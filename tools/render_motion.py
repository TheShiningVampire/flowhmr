"""Render a FlowHMR motion (.npz with poses/trans/betas) as a shaded SMPL-H mesh video, optionally
side by side with the source video. Pure numpy + PIL painter's renderer: no OpenGL, no browser.

    python tools/render_motion.py output/example/flowhmr_latest.npz out.mp4 \
        --video output/example/example_30fps.mp4 --size 720 --azimuth 35

Options: --fps (30), --size (720, output height), --workers (4), --azimuth / --elev / --dist (camera),
--device (cuda if available). Needs the V2M environment (torch, numpy, pillow) and ffmpeg on PATH.
Frames are rendered in parallel on the CPU; SMPL-H forward runs on --device with a 4 GB cap.
"""
import argparse, os, sys, subprocess, tempfile
import numpy as np, torch
from PIL import Image, ImageDraw
from concurrent.futures import ProcessPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def smplh_vertices(npz_path, device):
    os.chdir(REPO); sys.path.insert(0, REPO)
    from flowhmr.core.bodymodels.smpl_skeleton import SMPLMesh
    d = np.load(npz_path, allow_pickle=True)
    poses = torch.from_numpy(np.asarray(d["poses"], np.float32))
    trans = torch.from_numpy(np.asarray(d["trans"], np.float32))
    betas = torch.from_numpy(np.asarray(d["betas"], np.float32)).reshape(1, -1)
    T = poses.shape[0]; poses = poses.reshape(T, -1, 3)
    m = SMPLMesh().to(device)
    betas = betas[:, :m.shapedirs.shape[-1]].to(device)
    vs = []
    with torch.no_grad():
        for s in range(0, T, 64):
            e = min(T, s + 64)
            o = m({"poses": poses[s:e].to(device), "shapes": betas, "trans": trans[s:e].to(device)})
            vs.append(o["vertices"].float().cpu())
    return torch.cat(vs).numpy(), np.asarray(m.faces, np.int64)


def look_at(eye, target, up=(0, 1, 0)):
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    R = np.stack([r, u, -f])  # world -> camera rows
    return R, eye


def render_frame(args):
    V, F, size, R, eye, focal, floor_y, grid = args
    W = H = size
    img = Image.new("RGB", (W, H), (18, 20, 26))
    dr = ImageDraw.Draw(img)
    # ground grid
    for (a, b) in grid:
        pa = project(a, R, eye, focal, W, H); pb = project(b, R, eye, focal, W, H)
        if pa is not None and pb is not None:
            dr.line([pa, pb], fill=(45, 50, 62), width=1)
    # mesh: painter's algorithm, back to front
    Vc = (V - eye) @ R.T                      # camera coords, -z forward
    tri = Vc[F]                               # (N,3,3)
    depth = tri[:, :, 2].mean(1)
    order = np.argsort(depth)                 # most negative z = farthest -> draw first
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    light = np.array([0.3, 0.6, 0.75]); light /= np.linalg.norm(light)
    lam = np.clip(n @ light, 0, 1)
    base = np.array([110, 160, 235], float)
    cols = (base * (0.35 + 0.65 * lam[:, None])).astype(np.uint8)
    z = tri[:, :, 2]; xy = tri[:, :, :2]
    ok = (z < -0.05).all(1)
    px = W / 2 + focal * xy[:, :, 0] / (-z); py = H / 2 - focal * xy[:, :, 1] / (-z)
    back = n[:, 2] < 0                        # facing away from camera (camera looks along -z)
    for i in order:
        if not ok[i] or back[i]:
            continue
        dr.polygon([(px[i, 0], py[i, 0]), (px[i, 1], py[i, 1]), (px[i, 2], py[i, 2])],
                   fill=tuple(int(c) for c in cols[i]))
    return np.asarray(img)


def project(p, R, eye, focal, W, H):
    c = R @ (p - eye)
    if c[2] > -0.05:
        return None
    return (W / 2 + focal * c[0] / -c[2], H / 2 - focal * c[1] / -c[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz"); ap.add_argument("out")
    ap.add_argument("--video", default=None, help="source video to show on the left")
    ap.add_argument("--fps", type=int, default=30); ap.add_argument("--size", type=int, default=720)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--azimuth", type=float, default=35.0, help="camera azimuth in degrees around +y (0 = looking along -z toward the body front)")
    ap.add_argument("--elev", type=float, default=0.9, help="camera height above the body center in meters")
    ap.add_argument("--dist", type=float, default=None, help="camera distance in meters (default: auto from motion extent)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    a.npz, a.out = os.path.abspath(a.npz), os.path.abspath(a.out)
    if a.video: a.video = os.path.abspath(a.video)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    if a.device.startswith("cuda"):
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, 4 * 1024**3 / total), 0)
    V, F = smplh_vertices(a.npz, a.device)
    T = V.shape[0]
    floor_y = V[:, :, 1].min()
    center = V.reshape(-1, 3).mean(0); center[1] = floor_y + 0.9
    radius = a.dist or max(2.6, 1.3 * np.abs(V.reshape(-1, 3) - center).max() + 1.6)
    az = np.deg2rad(a.azimuth)
    eye = center + np.array([radius * np.sin(az), a.elev, radius * np.cos(az)])
    R, eye = look_at(eye, center)
    focal = a.size * 1.1
    g = []; ext = 4
    for k in range(-ext, ext + 1):
        g.append((center * [1, 0, 1] + [k, floor_y, -ext], center * [1, 0, 1] + [k, floor_y, ext]))
        g.append((center * [1, 0, 1] + [-ext, floor_y, k], center * [1, 0, 1] + [ext, floor_y, k]))
    jobs = [(V[t], F, a.size, R, eye, focal, floor_y, g) for t in range(T)]
    with ProcessPoolExecutor(a.workers) as ex:
        frames = list(ex.map(render_frame, jobs, chunksize=4))
    tmp = tempfile.mkdtemp(dir="/scratch" if os.path.isdir("/scratch") else None)
    for t, fr in enumerate(frames):
        Image.fromarray(fr).save(f"{tmp}/{t:05d}.png")
    mesh_mp4 = a.out if a.video is None else a.out + ".mesh.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "warning", "-framerate", str(a.fps), "-i", f"{tmp}/%05d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", mesh_mp4], check=True)
    if a.video is not None:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.video, "-i", mesh_mp4,
                        "-filter_complex",
                        f"[0:v]fps={a.fps},scale=-2:{a.size}[l];[1:v]scale=-2:{a.size}[r];[l][r]hstack=inputs=2,format=yuv420p",
                        "-c:v", "libx264", "-crf", "18", "-shortest", a.out], check=True)
    subprocess.run(["rm", "-rf", tmp])
    print("RENDER_OK", a.out, "frames", T)


if __name__ == "__main__":
    main()
