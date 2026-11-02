"""Whisper engine (Atticus.swift `// MARK: Whisper` section) compiled alone against a scratch ATTICUS_DIR."""
import functools
import hashlib
import http.server
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from tests import run_swift

ROOT = Path(__file__).resolve().parents[1]
TURBO = Path(os.environ.get("ATTICUS_WHISPER_TURBO", "~/Library/Application Support/whisper-models/ggml-large-v3-turbo-q5_0.bin")).expanduser()


def engine():
    source = (ROOT / "Atticus.swift").read_text()
    prefs = source[source.index("enum Prefs"):source.index("func applySharing(")]
    whisper = source[source.index("// MARK: Whisper"):source.index("// MARK: end Whisper")]
    return "import Cocoa\nimport AVFoundation\nimport CryptoKit\nlet controlKey = 1, optionKey = 2\n" + prefs + whisper


@unittest.skipUnless(sys.platform == "darwin", "macOS Swift regression")
class WhisperNativeTests(unittest.TestCase):
    def scratch(self):
        root = tempfile.mkdtemp(prefix="atticus-whisper-"); self.addCleanup(shutil.rmtree, root, True)
        p = patch.dict(os.environ, {"ATTICUS_DIR": root}); p.start(); self.addCleanup(p.stop)
        return Path(root)

    def test_wav_audio_and_catalog(self):
        self.scratch()
        run_swift(self, engine() + """
let w = Whisper.wav([0, 1, -1, 0.5])
precondition(w.count == 44 + 8)
precondition(String(decoding: w.prefix(4), as: UTF8.self) == "RIFF" && String(decoding: w[8..<16], as: UTF8.self) == "WAVEfmt ")
precondition(Array(w[4..<8]) == [44, 0, 0, 0])               // 36 + data bytes
precondition(Array(w[22..<28]) == [1, 0, 0x80, 0x3e, 0, 0])   // mono, 16000 Hz
precondition(Array(w[34..<36]) == [16, 0] && String(decoding: w[36..<40], as: UTF8.self) == "data" && Array(w[40..<44]) == [8, 0, 0, 0])
precondition(Array(w[44...]) == [0, 0, 0xff, 0x7f, 0x01, 0x80, 0xff, 0x3f])
precondition(Whisper.models.map(\\.id) == ["turbo", "large", "small"])
precondition(Whisper.model("large").file == "ggml-large-v3-q5_0.bin" && Whisper.model("nope").id == "turbo" && Whisper.model("").id == "turbo")
precondition(Whisper.dir.hasSuffix("/whisper") && Whisper.freePort()! > 1024 && Whisper.threads >= 1)
let a = WhisperAudio(), f = AVAudioFormat(standardFormatWithSampleRate: 16000, channels: 1)!
let b = AVAudioPCMBuffer(pcmFormat: f, frameCapacity: 8000)!; b.frameLength = 8000
a.append(b); a.append(b)
precondition(a.seconds == 1 && a.snapshot().count == 16000 && a.take().count == 16000 && a.seconds == 0)
let sem = DispatchSemaphore(value: 0)
Task {
    let quiet = try await Whisper.shared.transcribe([Float](repeating: 0.004, count: 32000), prompt: "")   // silence: no server
    let tap = try await Whisper.shared.transcribe([0.5, -0.5], prompt: "")
    precondition(quiet == "" && tap == "")
    sem.signal()
}
sem.wait()
print("ok")
""")

    def test_status_and_sha(self):
        root = self.scratch()
        (root / "whisper").mkdir()
        (root / "whisper" / "ggml-large-v3-q5_0.bin").write_bytes(b"x")
        blob = root / "blob"; blob.write_bytes(os.urandom(5_000_000))
        out = run_swift(self, engine() + f"""
precondition(Whisper.sha256("{blob}") == "{hashlib.sha256(blob.read_bytes()).hexdigest()}")
precondition(Whisper.sha256("/nonexistent") == nil)
precondition(!Whisper.shared.ready)   // turbo (the default) is missing
Prefs.set("whisperModel", "large")
precondition(Whisper.shared.ready == (Whisper.serverPath() != nil))
let s = Whisper.shared.status()
precondition(JSONSerialization.isValidJSONObject(s))
print(String(decoding: try! JSONSerialization.data(withJSONObject: s, options: .sortedKeys), as: UTF8.self))
""")
        import json
        s = json.loads(out)
        self.assertEqual(set(s), {"server", "model", "running", "error", "models"})
        self.assertEqual((s["model"], s["running"], s["error"]), ("large", False, None))
        self.assertEqual(s["server"], shutil.which("whisper-server") is not None or os.path.exists("/opt/homebrew/bin/whisper-server"))
        self.assertEqual([(m["id"], m["have"], m["progress"]) for m in s["models"]], [("turbo", False, None), ("large", True, None), ("small", False, None)])
        self.assertEqual(s["models"][0]["size"], "574 MB")
        self.assertEqual(s["models"][1]["name"], "Large v3")

    def test_download_verifies_sha_and_never_leaves_a_bad_file(self):
        root = self.scratch()
        site = root / "site"; site.mkdir()
        data = os.urandom(3_000_000); (site / "good.bin").write_bytes(data); (site / "bad.bin").write_bytes(data)
        class Quiet(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *a): pass
        handler = functools.partial(Quiet, directory=str(site))
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start(); self.addCleanup(httpd.shutdown)
        sha = hashlib.sha256(data).hexdigest()
        run_swift(self, engine() + f"""
Whisper.base = "http://127.0.0.1:{httpd.server_address[1]}/"
Whisper.models = [Whisper.Model(id: "good", name: "Good", file: "good.bin", sha256: "{sha}", bytes: {len(data)}),
                  Whisper.Model(id: "bad", name: "Bad", file: "bad.bin", sha256: "{'0' * 64}", bytes: {len(data)}),
                  Whisper.Model(id: "gone", name: "Gone", file: "missing.bin", sha256: "{sha}", bytes: 1)]
let w = Whisper.shared, sem = DispatchSemaphore(value: 0)
func fetch(_ id: String) -> [String: Any] {{
    w.download(id) {{ sem.signal() }}
    w.download(id) {{ preconditionFailure("second download of one id") }}
    var model: [String: Any] = [:]
    for _ in 0..<200 {{
        _ = sem.wait(timeout: .now() + 0.1)
        model = (w.status()["models"] as! [[String: Any]]).first {{ $0["id"] as? String == id }}!
        if model["progress"] is NSNull {{ return model }}
    }}
    preconditionFailure("download of \\(id) never finished")
}}
let fm = FileManager.default
precondition(fetch("good")["have"] as? Bool == true && w.status()["error"] is NSNull)
precondition(fm.contents(atPath: Whisper.dir + "/good.bin")?.count == {len(data)} && !fm.fileExists(atPath: Whisper.dir + "/good.bin.part"))
precondition(fetch("bad")["have"] as? Bool == false && w.status()["error"] as? String == "The download didn’t match. Try again.")
precondition(!fm.fileExists(atPath: Whisper.dir + "/bad.bin") && !fm.fileExists(atPath: Whisper.dir + "/bad.bin.part"))
precondition(fetch("gone")["have"] as? Bool == false && w.status()["error"] as? String == Whisper.offline)
w.delete("good"); w.delete("nope")
precondition(!fm.fileExists(atPath: Whisper.dir + "/good.bin") && w.status()["error"] is NSNull)
w.download("good") {{}}; w.cancel("good")
precondition(((w.status()["models"] as! [[String: Any]])[0]["progress"]) is NSNull)
print("ok")
""", timeout=60)

    def test_live_transcription(self):
        if not TURBO.exists() or not shutil.which("afconvert") or not os.path.exists("/opt/homebrew/bin/whisper-server") and not shutil.which("whisper-server"):
            self.skipTest("needs whisper-server, afconvert and the turbo model")
        root = self.scratch()
        (root / "whisper").mkdir(); (root / "whisper" / TURBO.name).symlink_to(TURBO)
        aiff, wav = root / "a.aiff", root / "a.wav"
        subprocess.run(["say", "-o", str(aiff), "The quick brown fox jumps over the lazy dog."], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEF32@16000", "-c", "1", str(aiff), str(wav)], check=True)
        out = run_swift(self, engine() + f"""
let file = try! AVAudioFile(forReading: URL(fileURLWithPath: "{wav}"))
let buf = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))!
try! file.read(into: buf)
precondition(file.processingFormat.sampleRate == 16000)
let audio = WhisperAudio(); audio.append(buf)
precondition(Whisper.shared.ready)
let sem = DispatchSemaphore(value: 0)
Task {{
    Whisper.shared.warm()
    print(try await Whisper.shared.transcribe(audio.take(), prompt: ""))
    precondition(Whisper.shared.status()["running"] as? Bool == true)
    Whisper.shared.stop(); sem.signal()
}}
sem.wait()
""", timeout=90)
        self.assertIn("fox", out.lower())

    def test_meeting_text_drops_invented_and_repeated_sentences(self):
        # Every example is a junk line from a real Vietnamese meeting transcript.
        self.scratch()
        run_swift(self, engine() + """
let m = MeetingWhisper(hints: "", heard: { _, _ in }, failed: { _ in })
func eq(_ a: String, _ b: String) { if a != b { print("got:", a, "want:", b); exit(1) } }
let t = Date()
eq(m.clean("Mình sẽ pick up những cái đó. Cảm ơn các bạn đã theo dõi và hẹn gặp lại.", at: t), "Mình sẽ pick up những cái đó.")
eq(m.clean("That's it for this video. Thank you for watching.", at: t), "")
eq(m.clean("If you enjoyed this video, please subscribe to my channel.", at: t), "")
eq(m.clean("Undertekster av Ai-Media", at: t), "")
eq(m.clean("Rồi với lại có cái nữa là bên test có against cái gì không. Rồi với lại có cái nữa là bên test có against cái gì không.", at: t), "Rồi với lại có cái nữa là bên test có against cái gì không.")
eq(m.clean("Rồi với lại có cái nữa là bên test có against cái gì không.", at: t.addingTimeInterval(10)), "")
eq(m.clean("share cho em xài 1 file này luôn rồi share cho em xài 1 file này luôn", at: t), "share cho em xài 1 file này luôn")
eq(m.clean("Mấy cái slide ở dưới sẽ dùng luôn.", at: t), "Mấy cái slide ở dưới sẽ dùng luôn.")
eq(m.clean("share cho em xài 1 file này luôn", at: t.addingTimeInterval(11)), "")
// real speech stays: short replies repeat, and a sentence said again a while later is new
// a sign-off stuck to real speech with no full stop between: only the sign-off goes
eq(m.clean("chị sẽ pin cái information đó thì mình sẽ pick up những Cảm ơn các bạn đã theo dõi và hẹn gặp lại.", at: t), "chị sẽ pin cái information đó thì mình sẽ pick up những")
eq(m.clean("mấy cái component này cũng cần thiết thôi Hẹn gặp lại các bạn trong những video tiếp theo.", at: t), "mấy cái component này cũng cần thiết thôi")
eq(m.clean("Please subscribe to my channel.", at: t), "")
eq(m.clean("Hãy subscribe cho kênh Ghiền Mì Gõ Để không bỏ lỡ những video hấp dẫn", at: t), "")
eq(m.clean("Dạ. Dạ.", at: t), "Dạ. Dạ.")
eq(m.clean("Bản 1.5 có sửa file config.json rồi nhé.", at: t), "Bản 1.5 có sửa file config.json rồi nhé.")
eq(m.clean("Thank you mọi người.", at: t), "Thank you mọi người.")
let n = MeetingWhisper(hints: "", heard: { _, _ in }, failed: { _ in })
eq(n.clean("Anh kiểm tra lại giúp em nhé.", at: t), "Anh kiểm tra lại giúp em nhé.")
eq(n.clean("Anh kiểm tra lại giúp em nhé.", at: t.addingTimeInterval(120)), "Anh kiểm tra lại giúp em nhé.")
// live words are cleaned but not remembered, or the final line would be dropped as a repeat of them
eq(n.clean("Bên test review qua cái file này. Thank you for watching.", at: t, remember: false), "Bên test review qua cái file này.")
eq(n.clean("Bên test review qua cái file này.", at: t), "Bên test review qua cái file này.")
print("ok")
""", timeout=30)

    def test_meeting_language_locks_when_two_long_lines_agree(self):
        self.scratch()
        run_swift(self, engine() + """
let l = MeetingLanguage()
l.note("english", seconds: 1); l.note("english", seconds: 1)
precondition(l.pinned == nil)                                   // short lines don't count: "you" is not a language
l.note("vietnamese", seconds: 5); precondition(l.pinned == nil)
l.note("spanish", seconds: 5); l.note("vietnamese", seconds: 5); precondition(l.pinned == nil)
l.note("vietnamese", seconds: 4); precondition(l.pinned == "vietnamese")
l.note("russian", seconds: 9); l.note("russian", seconds: 9); precondition(l.pinned == "vietnamese")   // locked for the meeting
Prefs.set("whisperLanguage", "auto"); precondition(l.request == "vietnamese")
Prefs.set("whisperLanguage", "de"); precondition(l.request == nil)   // a language picked in Settings always wins
print("ok")
""", timeout=30)

    def test_meeting_speech_check_hears_voices_not_noise(self):
        if not shutil.which("afconvert"):
            self.skipTest("needs afconvert")
        if "vi_VN" not in subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout:
            self.skipTest("needs the Vietnamese system voice")
        root = self.scratch()
        aiff, caf = root / "a.aiff", root / "a.caf"
        subprocess.run(["say", "-v", "Linh", "-o", str(aiff), "Xin chào mọi người, hôm nay chúng ta bàn về kế hoạch phát hành."], check=True)
        subprocess.run(["afconvert", "-f", "caff", "-d", "LEF32@16000", "-c", "1", str(aiff), str(caf)], check=True)
        data = caf.read_bytes(); (root / "v.f32").write_bytes(data[data.index(b"data") + 16:])
        run_swift(self, engine() + f"""
let voice = try! Data(contentsOf: URL(fileURLWithPath: "{root}/v.f32")).withUnsafeBytes {{ Array($0.bindMemory(to: Float.self)) }}
func room(_ secs: Int, _ a: Float) -> [Float] {{   // low-passed noise that swells, loud enough to pass the level gate
    var y: Float = 0
    return (0..<secs * 16000).map {{ i in y = 0.92 * y + 0.08 * Float.random(in: -1...1); return y * a * (0.5 + 0.5 * sin(Float(i) / 16000 * 2 * .pi * 0.4)) * 4 }}
}}
precondition(MeetingWhisper.isSpeech(voice), "voice")
precondition(MeetingWhisper.isSpeech(Array(voice[8000..<32000])), "1.5 s of voice")
precondition(!MeetingWhisper.isSpeech(room(5, 0.05)), "room")
precondition(!MeetingWhisper.isSpeech(room(20, 0.05)), "long room")
precondition(!MeetingWhisper.isSpeech(Array(voice[8000..<17600])), "0.6 s invents a whole sentence: too short to send")
print("ok")
""", timeout=30)

    def test_meeting_lines_close_at_the_last_pause(self):
        self.scratch()
        run_swift(self, engine() + """
func q(_ s: Double) -> [Float] { [Float](repeating: 0.001, count: Int(s * 16000)) }
func v(_ s: Double) -> [Float] { [Float](repeating: 0.2, count: Int(s * 16000)) }
precondition(MeetingWhisper.cut(v(3)) == nil)                                 // still talking
precondition(MeetingWhisper.cut(v(1) + q(0.5) + v(1)) == nil)                 // a breath is not a pause
let one = MeetingWhisper.cut(v(2) + q(0.8))!
precondition(one.speech && one.end == Int(2.8 * 16000) / 320 * 320)
let behind = MeetingWhisper.cut(v(2) + q(1) + v(2) + q(1) + v(1))!             // a slow pass missed two pauses: one pass catches up
precondition(behind.speech && behind.end == 6 * 16000)
let hush = MeetingWhisper.cut(q(2))!
precondition(!hush.speech && hush.end == 2 * 16000)
let click = MeetingWhisper.cut(q(1) + v(0.05) + q(1))!                         // a key click is not speech
precondition(!click.speech)
var noise = [Float](repeating: 0, count: 3 * 16000); for i in noise.indices { noise[i] = Float.random(in: -0.015...0.015) }
precondition(!MeetingWhisper.cut(noise)!.speech)                               // room noise never reaches Whisper
precondition(MeetingWhisper.hasSpeech(v(0.3)) && !MeetingWhisper.hasSpeech(v(0.1)) && !MeetingWhisper.hasSpeech(noise))
let a = WhisperAudio(), f = AVAudioFormat(standardFormatWithSampleRate: 16000, channels: 1)!
let b = AVAudioPCMBuffer(pcmFormat: f, frameCapacity: 1600)!; b.frameLength = 1600
a.append(b)
precondition(a.take(600).count == 600 && a.snapshot().count == 1000)
precondition(Whisper.withoutInvented("Hãy subscribe cho kênh La La School Để không bỏ lỡ những video hấp dẫn") == "" && Whisper.withoutInvented(" Hãy subscribe cho kênh Ghiền Mì Gõ") == "")
precondition(Whisper.withoutInvented("Chúng ta cần deploy bản hotfix lên production trước thứ sáu.") == "Chúng ta cần deploy bản hotfix lên production trước thứ sáu.")
precondition(Whisper.withoutInvented("Mình deploy thôi Cảm ơn các bạn đã theo dõi.") == "Mình deploy thôi")
print("ok")
""", timeout=30)

    def test_voice_asks_work_in_whisper_meetings(self):
        source = (ROOT / "Atticus.swift").read_text()
        self.assertIn("guard micAnalyzer != nil || micWhisper != nil, !demo", source)   # Whisper meetings have no SpeechAnalyzer

    def test_meeting_both_streams_mixed_english_noise_and_abrupt_end(self):
        # The real shape of a call: both streams share one server, Vietnamese with English terms, room noise, and audio that just stops.
        if not TURBO.exists() or not shutil.which("afconvert") or not os.path.exists("/opt/homebrew/bin/whisper-server") and not shutil.which("whisper-server"):
            self.skipTest("needs whisper-server, afconvert and the turbo model")
        voices = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
        if "vi_VN" not in voices or "Samantha" not in voices:
            self.skipTest("needs the Vietnamese and Samantha system voices")
        root = self.scratch()
        (root / "whisper").mkdir(); (root / "whisper" / TURBO.name).symlink_to(TURBO)
        def speak(name, parts):
            raw = bytearray()
            for i, (voice, text) in enumerate(parts):
                aiff, wav = root / f"{name}{i}.aiff", root / f"{name}{i}.caf"
                subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
                subprocess.run(["afconvert", "-f", "caff", "-d", "LEF32@16000", "-c", "1", str(aiff), str(wav)], check=True)
                data = wav.read_bytes(); raw += data[data.index(b"data") + 16:]
            return bytes(raw)
        V, E = "Linh", "Samantha"
        quiet = lambda s: b"\0\0\0\0" * int(s * 16000)
        call = speak("c1", [(V, "Chúng ta cần"), (E, "deploy"), (V, "bản"), (E, "hotfix"), (V, "lên"), (E, "production"), (V, "trước thứ sáu.")]) + quiet(1.5) \
            + speak("c2", [(V, "Anh kiểm tra lại"), (E, "API gateway"), (V, "giúp em nhé.")])   # stops dead: no silence after
        mic = quiet(2) + speak("m1", [(V, "Nếu lỗi thì mình"), (E, "rollback"), (V, "rồi mở một"), (E, "pull request"), (V, "mới.")]) + quiet(1.5)
        (root / "call.f32").write_bytes(call); (root / "mic.f32").write_bytes(mic)
        out = run_swift(self, engine() + f"""
import Foundation
func load(_ p: String) -> [Float] {{ try! Data(contentsOf: URL(fileURLWithPath: p)).withUnsafeBytes {{ Array($0.bindMemory(to: Float.self)) }} }}
var y: Float = 0
let noise: [Float] = (0..<16000 * 10).map {{ i in y = 0.92 * y + 0.08 * Float.random(in: -1...1); return y * 0.2 * (0.5 + 0.5 * sin(Float(i) / 16000 * 2 * .pi * 0.4)) }}
let callAudio = load("{root}/call.f32"), micAudio = load("{root}/mic.f32") + noise
var callFinals: [String] = [], micFinals: [String] = []
let c = MeetingWhisper(hints: "", heard: {{ t, final in if final && !t.isEmpty {{ callFinals.append(t) }} }}, failed: {{ print("failed", $0); exit(1) }})
let m = MeetingWhisper(hints: "", heard: {{ t, final in if final && !t.isEmpty {{ micFinals.append(t) }} }}, failed: {{ print("failed", $0); exit(1) }})
Whisper.shared.warm()
func feed(_ w: MeetingWhisper, _ all: [Float]) {{
    var at = 0
    Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) {{ t in
        guard at < all.count else {{ t.invalidate(); return }}
        let piece = Array(all[at..<min(all.count, at + 1600)]); at += piece.count
        let b = AVAudioPCMBuffer(pcmFormat: AVAudioFormat(standardFormatWithSampleRate: 16000, channels: 1)!, frameCapacity: AVAudioFrameCount(piece.count))!
        b.frameLength = AVAudioFrameCount(piece.count); piece.withUnsafeBufferPointer {{ b.floatChannelData![0].update(from: $0.baseAddress!, count: piece.count) }}
        w.audio.append(b)
    }}
}}
feed(c, callAudio); feed(m, micAudio)
Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) {{ _ in c.tick(); m.tick() }}
DispatchQueue.main.asyncAfter(deadline: .now() + {len(mic) / 64000 + 10 + 15}) {{
    print("call:", callFinals.joined(separator: " | ")); print("mic:", micFinals.joined(separator: " | "))
    print("open:", c.audio.seconds, m.audio.seconds); Whisper.shared.stop(); exit(0)
}}
RunLoop.main.run()
""", timeout=120)
        lines = dict(line.split(":", 1) for line in out.splitlines() if ":" in line)
        self.assertEqual(lines["open"].split(), ["0.0", "0.0"], out)   # nothing stranded, even after audio stops dead
        self.assertIn("production", lines["call"].lower(), out)
        self.assertIn("api", lines["call"].lower(), out)               # the abrupt last line closed
        self.assertIn("rollback", lines["mic"].lower(), out)
        self.assertNotIn("subscribe", lines["mic"].lower(), out)       # noise invented nothing
        self.assertEqual(lines["mic"].count("|"), 0, out)              # the noise made no line of its own

    def test_meeting_stream_hears_vietnamese(self):
        # Meetings on Whisper: audio arrives in 0.1 s pieces like the capture stream; live words first, a pause closes the line.
        if not TURBO.exists() or not shutil.which("afconvert") or not os.path.exists("/opt/homebrew/bin/whisper-server") and not shutil.which("whisper-server"):
            self.skipTest("needs whisper-server, afconvert and the turbo model")
        if "vi_VN" not in subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout:
            self.skipTest("needs the Vietnamese system voice")
        root = self.scratch()
        (root / "whisper").mkdir(); (root / "whisper" / TURBO.name).symlink_to(TURBO)
        aiff, wav = root / "a.aiff", root / "a.wav"
        subprocess.run(["say", "-v", "Linh", "-o", str(aiff), "Xin chào mọi người. Hôm nay chúng ta sẽ bàn về kế hoạch phát hành phiên bản mới vào thứ sáu."], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEF32@16000", "-c", "1", str(aiff), str(wav)], check=True)
        out = run_swift(self, engine() + f"""
let file = try! AVAudioFile(forReading: URL(fileURLWithPath: "{wav}"))
let buf = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))!
try! file.read(into: buf)
let all = Array(UnsafeBufferPointer(start: buf.floatChannelData![0], count: Int(buf.frameLength))) + [Float](repeating: 0, count: 24000)
var live = 0, finals: [String] = []
let m = MeetingWhisper(hints: "", heard: {{ text, final in if final {{ if !text.isEmpty {{ finals.append(text) }} }} else {{ live += 1 }} }}, failed: {{ print("failed", $0); exit(1) }})
Whisper.shared.warm()
var at = 0
Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) {{ _ in   // real time, then a pause
    guard at < all.count else {{ return }}
    let piece = Array(all[at..<min(all.count, at + 1600)]); at += piece.count
    let b = AVAudioPCMBuffer(pcmFormat: AVAudioFormat(standardFormatWithSampleRate: 16000, channels: 1)!, frameCapacity: AVAudioFrameCount(piece.count))!
    b.frameLength = AVAudioFrameCount(piece.count); piece.withUnsafeBufferPointer {{ b.floatChannelData![0].update(from: $0.baseAddress!, count: piece.count) }}
    m.audio.append(b)
}}
Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) {{ _ in m.tick() }}
Timer.scheduledTimer(withTimeInterval: 1, repeats: true) {{ _ in
    guard at >= all.count, !finals.isEmpty else {{ return }}
    print("live", live); print(finals.joined(separator: " | ")); Whisper.shared.stop(); exit(0)
}}
DispatchQueue.main.asyncAfter(deadline: .now() + 60) {{ print("timed out", live, finals); exit(1) }}
RunLoop.main.run()
""", timeout=90)
        self.assertRegex(out, r"live [1-9]")
        self.assertIn("chào", out.lower())
        self.assertIn("phiên bản", out.lower())


if __name__ == "__main__":
    unittest.main()
