"""Assemble the FlowHMR explainer: PIL slides + piper narration + demo clips -> one mp4.

  python make_video.py scenes.py out.mp4 [--work DIR] [--voice VOICE.onnx]
scenes.py must define SCENES: list of dicts with keys
  kind: "slide" | "clip"
  title, bullets (list[str]), footnote (slide); diagram: "pipeline" (slide, optional)
  video (clip path), caption (clip)
  narration: str (text spoken over the scene)
"""
import argparse, os, subprocess, wave, importlib.util, textwrap
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
BG = (16, 18, 24); FG = (236, 238, 242); MUTED = (150, 156, 170); ACCENT = (96, 165, 250); ACCENT2 = (52, 211, 153)
FONT_DIR = "/usr/share/fonts/dejavu-sans-fonts"


def font(size, bold=False):
    for name in (["DejaVuSans-Bold.ttf", "DejaVuSansCondensed-Bold.ttf"] if bold else ["DejaVuSans.ttf", "DejaVuSansCondensed.ttf"]):
        p = os.path.join(FONT_DIR, name)
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def draw_pipeline(dr, y0):
    boxes = [("Input video", "monocular, any fps"), ("YOLOX", "detect + track person"),
             ("VGGT-Omega", "camera rotation per frame"), ("SAM 3D Body", "3072-d body tokens"),
             ("FlowHMR", "flow matching, 20 ODE steps"), ("SMPL-H motion", "poses (T,52,3), trans (T,3)")]
    n = len(boxes); margin = 90; gap = 28
    bw = (W - 2 * margin - gap * (n - 1)) // n; bh = 150
    f1, f2 = font(28, True), font(21)
    for i, (t, s) in enumerate(boxes):
        x = margin + i * (bw + gap)
        col = ACCENT if i == 4 else (40, 44, 56)
        dr.rounded_rectangle([x, y0, x + bw, y0 + bh], radius=16, fill=col, outline=(70, 76, 92), width=2)
        tc = (10, 12, 16) if i == 4 else FG
        dr.text((x + bw / 2, y0 + 50), t, font=f1, fill=tc, anchor="mm")
        for k, line in enumerate(textwrap.wrap(s, 20)):
            dr.text((x + bw / 2, y0 + 92 + k * 26), line, font=f2, fill=tc if i == 4 else MUTED, anchor="mm")
        if i < n - 1:
            ax = x + bw + gap / 2
            dr.line([(x + bw + 4, y0 + bh / 2), (x + bw + gap - 4, y0 + bh / 2)], fill=MUTED, width=4)
            dr.polygon([(x + bw + gap - 2, y0 + bh / 2), (x + bw + gap - 14, y0 + bh / 2 - 8), (x + bw + gap - 14, y0 + bh / 2 + 8)], fill=MUTED)
    # physics reward row (training only)
    y1 = y0 + bh + 70
    dr.rounded_rectangle([margin + 3 * (bw + gap), y1, margin + 5 * (bw + gap) + bw, y1 + 110], radius=16, fill=(30, 48, 44), outline=ACCENT2, width=2)
    cx = margin + 3 * (bw + gap) + (2 * (bw + gap) + bw) / 2
    dr.text((cx, y1 + 36), "GRPO post-training (training only)", font=font(28, True), fill=ACCENT2, anchor="mm")
    dr.text((cx, y1 + 78), "reward = video fidelity + PHC+ controller can track the motion in Isaac Gym", font=font(20), fill=FG, anchor="mm")


def render_slide(sc, path):
    img = Image.new("RGB", (W, H), BG); dr = ImageDraw.Draw(img)
    dr.rectangle([0, 0, 18, H], fill=ACCENT)
    dr.text((90, 70), sc["title"], font=font(60, True), fill=FG)
    if sc.get("subtitle"):
        dr.text((90, 150), sc["subtitle"], font=font(28), fill=MUTED)
    y = 250
    if sc.get("diagram") == "pipeline":
        draw_pipeline(dr, 300); y = 720
    fb = font(sc.get("bullet_size", 36)); lh = int(sc.get("bullet_size", 36) * 1.55)
    for b in sc.get("bullets", []):
        lines = textwrap.wrap(b, sc.get("wrap", 78))
        dr.ellipse([96, y + lh * 0.32, 112, y + lh * 0.32 + 16], fill=ACCENT)
        for k, line in enumerate(lines):
            dr.text((140, y), line, font=fb, fill=FG); y += lh
        y += int(lh * 0.35)
    if sc.get("table"):
        ft = font(32); fth = font(32, True); x0 = 140; cw = sc.get("colw", [520, 300, 300, 400])
        for r, row in enumerate(sc["table"]):
            for c, cell in enumerate(row):
                dr.text((x0 + sum(cw[:c]), y), str(cell), font=fth if r == 0 else ft, fill=ACCENT if r == 0 else FG)
            y += 52
            if r == 0:
                dr.line([(x0, y - 6), (x0 + sum(cw), y - 6)], fill=(70, 76, 92), width=2)
    if sc.get("footnote"):
        dr.text((90, H - 70), sc["footnote"], font=font(24), fill=MUTED)
    img.save(path)


def tts(text, wav, voice):
    subprocess.run(["piper", "-m", voice, "-f", wav, "--length-scale", "1.05", "--sentence-silence", "0.35"],
                   input=text.encode(), check=True, stderr=subprocess.DEVNULL)
    with wave.open(wav) as w:
        return w.getnframes() / w.getframerate()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("scenes"); ap.add_argument("out")
    ap.add_argument("--work", default=None); ap.add_argument("--voice", default="en_US-lessac-medium.onnx")
    a = ap.parse_args()
    work = a.work or (a.out + ".work"); os.makedirs(work, exist_ok=True)
    spec = importlib.util.spec_from_file_location("scenes", a.scenes); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    segs = []
    for i, sc in enumerate(m.SCENES):
        base = os.path.join(work, f"s{i:02d}"); wav = base + ".wav"; seg = base + ".mp4"
        dur = tts(sc["narration"], wav, a.voice) + 1.0
        common = ["-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30",
                  "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2", "-shortest"]
        if sc["kind"] == "slide":
            png = base + ".png"; render_slide(sc, png)
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-framerate", "30", "-i", png,
                   "-i", wav, "-af", "apad", "-t", f"{dur:.2f}", "-vf", f"scale={W}:{H}"] + common + [seg]
        else:
            cap = sc.get("caption", "").replace(":", r"\:").replace("'", r"\'")
            vf = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x101218,"
                  f"drawtext=fontfile={FONT_DIR}/DejaVuSans-Bold.ttf:text='{cap}':fontcolor=white:fontsize=34:x=(w-text_w)/2:y=h-80:box=1:boxcolor=0x101218AA:boxborderw=14")
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-stream_loop", "-1", "-i", sc["video"], "-i", wav, "-af", "apad",
                   "-t", f"{max(dur, sc.get('min_dur', 0)):.2f}", "-vf", vf] + common + [seg]
        subprocess.run(cmd, check=True)
        segs.append(seg); print(f"scene {i:02d} {sc['kind']:5s} {dur:5.1f}s  {sc.get('title', sc.get('caption',''))[:50]}")
    lst = os.path.join(work, "list.txt")
    with open(lst, "w") as f:
        for s in segs:
            f.write(f"file '{s}'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", a.out], check=True)
    total = sum(float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", s])) for s in segs)
    print(f"VIDEO_OK {a.out} {total/60:.1f} min")


if __name__ == "__main__":
    main()
