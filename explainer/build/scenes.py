"""Scene list for the FlowHMR explainer. Numbers marked TODO are filled from the real run."""
import os, json
STATS = {}
sp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_stats.json")
if os.path.exists(sp):
    STATS = json.load(open(sp))
S = lambda k, d: STATS.get(k, d)
DEMO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "demo_example_side_by_side.mp4")
SMOKE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "smoke_test_render.mp4")

SCENES = [
 dict(kind="slide", title="FlowHMR, explained by running it",
      subtitle="Physically plausible motion capture from a single video  |  arXiv 2610.03691  |  github.com/TheShiningVampire/flowhmr",
      bullets=["What the paper claims, and why it matters",
               "What the pipeline actually does when you run it",
               "What it took to run it on the Babel cluster, and what I changed"],
      footnote="Made by Claude Code on Babel for Vinit Awale, October 2026",
      narration="This is FlowHMR, a method for physically plausible motion capture from a single video. "
                "I forked the official repository, set it up on the Babel cluster, ran the bundled demo, and "
                "built this explainer from what happened. We will cover the idea, the pipeline, the run, and the changes I made."),
 dict(kind="slide", title="The problem",
      bullets=["Goal: from one ordinary video, recover the full 3D motion of the person, including where they move in the world",
               "Depth is ambiguous from one camera, so regression models average over the possibilities and produce dull, floaty motion",
               "Those motions often break physics: feet slide, bodies lean through their support, and a simulated character cannot copy them",
               "FlowHMR measures this directly: can a physics-based controller track the recovered motion in simulation?"],
      narration="Here is the problem. Given one ordinary video, we want the full three D motion of the person, including how they move through the world. "
                "A single camera cannot see depth, so networks that regress motion directly tend to average over the possibilities. "
                "The result is dull, floaty motion that often breaks physics: feet slide and bodies lean through their support. "
                "FlowHMR measures this directly, by asking whether a physics based controller can track the recovered motion in simulation."),
 dict(kind="slide", title="The idea: generate, then prefer the physical ones",
      bullets=["Treat motion capture as video-conditioned motion generation with a flow-matching model, not regression",
               "Backbone: an MMDiT transformer from HY-Motion, 18 layers, 0.46 billion parameters, 20 Euler ODE steps at inference",
               "Pretraining gives diverse candidates; not all are faithful to the video or physically trackable",
               "GRPO post-training with two rewards: fidelity to the video, and whether the PHC+ controller can track the motion in Isaac Gym",
               "Result: tracking success 78.8% for the base model, 82.5% after GRPO, versus 62.8% for GVHMR, the strongest baseline"],
      bullet_size=34,
      narration="The idea has two parts. First, treat motion capture as video conditioned motion generation. A flow matching transformer, "
                "built on the HY-Motion backbone with eighteen layers and about half a billion parameters, generates motion from video features "
                "in twenty ODE steps. Second, since not every generated motion is faithful or physical, the model is post trained with GRPO, "
                "group relative policy optimization, using two rewards: fidelity to the video, and whether the PHC plus physics controller can "
                "track the motion in Isaac Gym. That lifts tracking success from seventy nine percent to eighty two and a half, against sixty three "
                "percent for the strongest baseline, GVHMR."),
 dict(kind="slide", title="What runs when you call the demo", diagram="pipeline",
      bullets=["Each stage caches its output next to the video, so re-running with another checkpoint only repeats generation",
               "Optional post-processing: temporal smoothing plus foot-contact and ground-contact inverse kinematics"],
      bullet_size=32,
      narration="This is the inference pipeline that runs on a raw video. The video is transcoded to thirty frames per second. "
                "YOLOX detects and tracks the person. VGGT Omega estimates the camera rotation for every frame. SAM 3D Body turns each "
                "frame into a three thousand seventy two dimensional body token. FlowHMR then generates the motion as SMPL H parameters: "
                "fifty two joint rotations and a root translation per frame. The physics controller is used only during training, as a reward. "
                "Each stage caches its output, so a second run with another checkpoint repeats only the generation step."),
 dict(kind="slide", title="Setting it up on Babel",
      bullets=["Forked flowhmr/flowhmr to TheShiningVampire/flowhmr; cloned with four shallow submodules: PHC, VGGT-Omega, YOLOX, SAM 3D Body",
               "Python 3.10 environment with uv under /data/user_data: torch 2.5.1 with CUDA 11.8, built in under two minutes",
               "Public assets: FlowHMR checkpoint (1.8 GB), YOLOX-L weights, the GVHMR joint regressor",
               "Gated assets that needed a human: SMPL-H body model from the MANO site, plus Hugging Face approval for VGGT-Omega and SAM 3D Body",
               "Everything ran on one shared RTX 6000 Ada, next to five live training jobs, with a 14 GB per-process memory cap"],
      bullet_size=33,
      narration="Setting it up on the cluster was mostly smooth. I forked the repository, cloned it with its four submodules, and built a "
                "Python three point ten environment with uv in under two minutes. The public checkpoints downloaded without trouble. "
                "The slow part was licensing: the SMPL H body model had to be fetched by Vinit from the MANO site, and the two Meta "
                "perception models are gated on Hugging Face and needed manual approval. Everything then ran on a single shared GPU, "
                "next to five live training jobs, with a fourteen gigabyte memory cap so it could never starve them."),
 dict(kind="slide", title="First test: the generator alone",
      bullets=["Lite mode loads only FlowHMR and the SMPL-H body model, skipping the perception models",
               "150 frames of motion from random body tokens: 4.6 seconds of sampling, peak 1.9 GB of GPU memory",
               "Confirms the checkpoint, the body model, the ODE sampler, and the npz export before touching the heavy models",
               "The motion below is from random tokens, so it is nonsense on purpose; it validates the rendering path"],
      narration="Before running the heavy perception models, I tested the generator alone. Lite mode loads only FlowHMR and the body model. "
                "Generating one hundred fifty frames from random tokens took under five seconds and under two gigabytes of GPU memory. "
                "That confirmed the checkpoint, the body model, the sampler and the export. The motion you are about to see comes from "
                "random tokens, so it is meaningless on purpose. It only validates the rendering path I wrote for this video."),
]
if os.path.exists(SMOKE):
    SCENES.append(dict(kind="clip", video=SMOKE, caption="Smoke test: motion from random tokens, not a real result", min_dur=6,
                       narration="A shaded SMPL H mesh rendered with a small software renderer, no OpenGL required. Random tokens, random motion."))
if os.path.exists(DEMO):
    SCENES += [
     dict(kind="slide", title="The real run: the bundled example video",
          bullets=[f"Input: {S('in_frames', 'TODO')} frames at {S('in_fps', 'TODO')} fps, {S('in_res', 'TODO')}, transcoded to 30 fps",
                   f"Detection and tracking: {S('t_detect', 'TODO')} s;  camera estimation: {S('t_camera', 'TODO')} s;  body features: {S('t_feat', 'TODO')} s",
                   f"FlowHMR generation: {S('t_gen', 'TODO')} s for {S('out_frames', 'TODO')} frames;  whole pipeline: {S('t_total', 'TODO')} s",
                   f"Peak GPU memory: {S('peak_gb', 'TODO')} GB with camera estimation on every {S('vggt_interval', 'TODO')}th frame",
                   "Output: output/example/flowhmr_latest.npz with poses (T, 52, 3), trans (T, 3), betas, foot contacts"],
          bullet_size=33,
          narration=S("narr_run", "Now the real run on the bundled example video. TODO narration from stats.")),
     dict(kind="clip", video=DEMO, caption="Left: input video.   Right: FlowHMR output as SMPL-H mesh", min_dur=10,
          narration=S("narr_demo", "On the left is the input video. On the right is the motion FlowHMR generated, rendered as the SMPL H mesh in world space. "
                      "Watch the feet and the root trajectory: this is the part the physics reward was trained to fix.")),
    ]
SCENES += [
 dict(kind="slide", title="What I changed in the fork",
      bullets=S("changes", ["TODO: list of branch changes"]),
      footnote="Branch and pull request links are in the chat summary",
      bullet_size=34,
      narration=S("narr_changes", "TODO narration about changes.")),
 dict(kind="slide", title="Takeaways",
      bullets=["The method is sound and the inference code is clean: one script, cached stages, clear error messages for missing assets",
               "Setup cost is dominated by licenses, not code: three gated downloads, two of them waiting on human approval",
               "Inference fits on a shared GPU; the physics side (PHC+, Isaac Gym, GRPO) is a second environment and a much bigger lift",
               "Worth your time if you need world-grounded, trackable human motion from video, for example as reference motion for humanoid control"],
      bullet_size=34,
      narration="Takeaways. The method is sound and the inference code is clean: a single script, cached stages, and clear errors when an asset is missing. "
                "The setup cost is dominated by licenses rather than code. Inference fits comfortably on a shared GPU, while the physics side, "
                "with PHC plus, Isaac Gym and GRPO, is a separate environment and a much bigger lift. It is worth your time if you need world "
                "grounded, physically trackable human motion from video, for example as reference motion for humanoid control. That is FlowHMR."),
]
