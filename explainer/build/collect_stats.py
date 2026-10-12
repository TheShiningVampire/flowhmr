"""Parse the timestamped demo log into run_stats.json (numbers + narration) for scenes.py.
  python collect_stats.py ../logs/demo_run.log run_stats.json [--changes changes.txt]
"""
import re, sys, json, datetime as dt, subprocess, argparse
ap = argparse.ArgumentParser(); ap.add_argument("log"); ap.add_argument("out"); ap.add_argument("--changes", default=None)
a = ap.parse_args()
lines = open(a.log, errors="replace").read().splitlines()
def ts(line):
    m = re.match(r"(\d\d):(\d\d):(\d\d) ", line)
    return None if not m else int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])
marks = {}
for ln in lines:
    for key, pat in [("detect", "=== Step 1"), ("camera", "=== Step 2"), ("feat", "=== Step 3"), ("gen", "=== Step 4"), ("done", r"\[ok\]")]:
        if re.search(pat, ln) and key not in marks and ts(ln) is not None:
            marks[key] = ts(ln)
first = next((ts(l) for l in lines if ts(l) is not None), None)
S = {}
if all(k in marks for k in ("detect", "camera", "feat", "gen", "done")):
    S.update(t_detect=marks["camera"] - marks["detect"], t_camera=marks["feat"] - marks["camera"],
             t_feat=marks["gen"] - marks["feat"], t_gen=marks["done"] - marks["gen"], t_load=marks["detect"] - first)
m = re.search(r"wall=(\d+)s peak_gpu_mib=\s*(\d+)", "\n".join(lines))
if m:
    S["t_total"] = int(m[1]); S["peak_gb"] = round(int(m[2]) / 1024, 1)
m = re.search(r"\[gen\] T=(\d+)", "\n".join(lines))
if m: S["out_frames"] = int(m[1])
m = re.search(r"interval=(\d+)", "\n".join(lines))
if m: S["vggt_interval"] = int(m[1])
m = re.search(r"GPU memory cap: ([\d.]+) GB", "\n".join(lines))
if m: S["cap_gb"] = float(m[1])
S.update(in_frames=121, in_fps=24, in_res="1920x1080")
if a.changes:
    S["changes"] = [l.strip() for l in open(a.changes) if l.strip()]
g = lambda k: S.get(k, "unknown")
S["narr_run"] = (f"Now the real run on the bundled example: a five second, nineteen twenty by ten eighty clip at twenty four frames per second, "
                 f"transcoded to thirty. Loading all models took {g('t_load')} seconds. Detection and tracking took {g('t_detect')} seconds, "
                 f"camera estimation {g('t_camera')} seconds with VGGT Omega on every {g('vggt_interval')}th frame, body feature extraction "
                 f"{g('t_feat')} seconds, and FlowHMR generation itself about {max(1, S.get('t_gen', 1))} second{'s' if S.get('t_gen', 1) != 1 else ''} for {g('out_frames')} frames. "
                 f"The whole script finished in {g('t_total')} seconds, with a peak of {g('peak_gb')} gigabytes of GPU memory, "
                 f"well under the {g('cap_gb')} gigabyte cap, on a card that was already running five other jobs.")
S["narr_demo"] = ("On the left is the input video. On the right is the motion FlowHMR generated, rendered as the SMPL H mesh in world "
                  "space with my offline renderer. Watch the feet and the root trajectory: this is exactly the part the physics reward "
                  "was trained to fix, and it is what the PHC plus controller would be asked to track.")
S["narr_changes"] = ("Everything I changed lives on one branch in the fork. First, a memory cap flag for the demo script, so it can share a "
                     "GPU with other jobs on a cluster. Second, an offline renderer that turns the output motion into a shaded mesh video, "
                     "or a side by side with the source, without a browser or OpenGL. Third, a gitignore fix for output directories that are "
                     "symlinks to bulk storage. And a short note in the docs on running on shared GPUs. The branch is pushed to the fork.")
json.dump(S, open(a.out, "w"), indent=1); print(json.dumps(S, indent=1)[:1500])
