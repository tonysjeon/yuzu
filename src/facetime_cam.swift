import AVFoundation
import CoreMedia
import CoreVideo
import Foundation

/// Capture the Mac's built-in wide-angle camera only.
/// Never discovers Continuity / iPhone / external devices.
final class Sink: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let lock = NSLock()
    private var latest: Data?
    private(set) var width = 0
    private(set) var height = 0

    func take() -> (Data, Int, Int)? {
        lock.lock()
        defer { lock.unlock() }
        guard let latest else { return nil }
        self.latest = nil
        return (latest, width, height)
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let image = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        CVPixelBufferLockBaseAddress(image, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(image, .readOnly) }
        let w = CVPixelBufferGetWidth(image)
        let h = CVPixelBufferGetHeight(image)
        let stride = CVPixelBufferGetBytesPerRow(image)
        guard let base = CVPixelBufferGetBaseAddress(image) else { return }
        let rowBytes = w * 4
        var packed = Data(count: rowBytes * h)
        packed.withUnsafeMutableBytes { dest in
            guard let dst = dest.baseAddress else { return }
            if stride == rowBytes {
                memcpy(dst, base, rowBytes * h)
            } else {
                for row in 0..<h {
                    memcpy(
                        dst.advanced(by: row * rowBytes),
                        base.advanced(by: row * stride),
                        rowBytes
                    )
                }
            }
        }
        lock.lock()
        latest = packed
        width = w
        height = h
        lock.unlock()
    }
}

func fail(_ message: String) -> Never {
    fputs(message + "\n", stderr)
    exit(1)
}

let wantW = CommandLine.arguments.count > 1 ? (Int(CommandLine.arguments[1]) ?? 1280) : 1280
let wantH = CommandLine.arguments.count > 2 ? (Int(CommandLine.arguments[2]) ?? 720) : 720

guard let device = AVCaptureDevice.default(
    .builtInWideAngleCamera,
    for: .video,
    position: .unspecified
) else {
    fail("No built-in FaceTime camera found.")
}

do {
    try device.lockForConfiguration()
    var best: AVCaptureDevice.Format?
    var bestScore = Int.max
    for format in device.formats {
        let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
        let score = abs(Int(dims.width) - wantW) + abs(Int(dims.height) - wantH)
        if score < bestScore {
            bestScore = score
            best = format
        }
    }
    if let best {
        device.activeFormat = best
    }
    device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30)
    device.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 30)
    device.unlockForConfiguration()
} catch {
    fail("Could not configure the built-in camera: \(error)")
}

let session = AVCaptureSession()
session.beginConfiguration()
if session.canSetSessionPreset(.hd1280x720) {
    session.sessionPreset = .hd1280x720
}

let input: AVCaptureDeviceInput
do {
    input = try AVCaptureDeviceInput(device: device)
} catch {
    fail("Could not open the built-in camera: \(error)")
}
guard session.canAddInput(input) else {
    fail("Could not attach the built-in camera.")
}
session.addInput(input)

let output = AVCaptureVideoDataOutput()
output.alwaysDiscardsLateVideoFrames = true
output.videoSettings = [
    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
]
let sink = Sink()
output.setSampleBufferDelegate(sink, queue: DispatchQueue(label: "yuzu.facetime"))
guard session.canAddOutput(output) else {
    fail("Could not attach the camera output.")
}
session.addOutput(output)
session.commitConfiguration()
session.startRunning()

var headerW = 0
var headerH = 0
for _ in 0..<80 {
    if let frame = sink.take() {
        headerW = frame.1
        headerH = frame.2
        break
    }
    Thread.sleep(forTimeInterval: 0.05)
}
guard headerW > 0, headerH > 0 else {
    fail("Built-in camera produced no frames.")
}

let header: [String: Any] = [
    "width": headerW,
    "height": headerH,
    "name": device.localizedName,
]
guard var payload = try? JSONSerialization.data(withJSONObject: header) else {
    fail("Could not encode camera header.")
}
payload.append(0x0A)
FileHandle.standardOutput.write(payload)

let stdout = FileHandle.standardOutput
while true {
    if let (frame, _, _) = sink.take() {
        var n = UInt32(frame.count).bigEndian
        withUnsafeBytes(of: &n) { stdout.write(Data($0)) }
        stdout.write(frame)
    } else {
        Thread.sleep(forTimeInterval: 0.004)
    }
}
