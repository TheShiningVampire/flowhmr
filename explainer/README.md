# FlowHMR explainer

Videos made while setting up and running this fork on the CMU Babel cluster (October 2026).

| File | What |
|---|---|
| `flowhmr_explainer.mp4` | 5.5 min narrated explainer: the paper's idea, the inference pipeline, the cluster setup, a smoke test, the real run with timings, the changes on this branch, takeaways. |
| `demo_example_side_by_side.mp4` | The real run on `assets/example/example.mp4`: input (left) and the `flowhmr_latest` output rendered as an SMPL-H mesh with `tools/render_motion.py` (right). |
| `smoke_test_render.mp4` | Lite-mode smoke test: motion generated from random body tokens, rendered the same way. Not a real result; it only validates the rendering path. |

## Measured on the run

Shared RTX 6000 Ada (five other jobs on the card), `--max_gpu_mem_gb 14 --vggt_interval 10`:

| Stage | Time |
|---|---|
| Model loading | 20 s |
| YOLOX detection + tracking | 7 s |
| VGGT-Omega camera (every 10th frame) | 18 s |
| SAM 3D Body features, 151 frames | 460 s |
| FlowHMR generation | ~1 s |
| Whole script | 509 s |
| Peak GPU memory | 10.2 GB |

## Rebuilding the video

`build/` holds the generator: `make_video.py` draws the slides with Pillow, narrates each scene with
[piper](https://github.com/rhasspy/piper) (voice `en_US-lessac-medium`), and assembles slides and clips with ffmpeg.
`scenes.py` is the script and narration; `collect_stats.py` turns a timestamped demo log into `run_stats.json`.

```bash
pip install piper-tts pillow numpy   # plus ffmpeg on PATH
python build/make_video.py build/scenes.py flowhmr_explainer.mp4 --voice <path to en_US-lessac-medium.onnx>
```

`scenes.py` looks for the demo clips next to itself; adjust the `DEMO` / `SMOKE` paths if you move them.
