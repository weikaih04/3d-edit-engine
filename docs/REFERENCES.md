# References

## Frozen models the engine runs on

| Model | Role here | Link |
|---|---|---|
| **TRELLIS.2-4B** (Microsoft) | 3D engine: texture-given-shape re-sampling (E1a/b/c), full image-to-3D (E5) | github.com/microsoft/TRELLIS.2 · HF `microsoft/TRELLIS.2-4B` |
| **P3-SAM** (Tencent Hunyuan3D-Part) | native 3D part segmentation → `face_ids` for every part op | github.com/Tencent-Hunyuan/Hunyuan3D-Part · HF `tencent/Hunyuan3D-Part` |
| **X-Part** (same repo) | in-context part re-synthesis (E4 chain; E2/E8 candidate) | same as above |
| **Qwen-Image-Edit-2511** | instruction-driven 2D view editing (E1a/E1b/E1c/E5 conditions) | HF `Qwen/Qwen-Image-Edit-2511` |
| **Qwen3.6-27B** (vLLM) | part naming, dual-blind verification, upstream captions/judge | HF `Qwen/Qwen3.6-27B` |
| DINOv3 ViT-L | TRELLIS.2 image conditioning backbone | HF `facebook/dinov3-vitl16-pretrain-lvd1689m` (gated) |
| TRELLIS (v1) image-large | sparse-structure decoder required by TRELLIS.2's full pipeline; base model of Nano3D/VoxHammer | github.com/microsoft/TRELLIS |

## 3D-editing papers that shaped the task matrix (V1 plan lit review)

Cloned reference code lives in `<workdir>/reference/` (not committed here).

| Work | What we took / rejected |
|---|---|
| **Nano3D** — *Training-Free 3D Editing Without Masks* (ICLR'26), arXiv:2510.15019, github.com/JAMESYJL/Nano3D | FlowEdit-on-TRELLIS reference implementation (~50-line velocity-difference sampler on the dense SS latent; n_avg=5, st_step=12, src/tar cfg 1.5/5.5) — the phase-2 port target for E1c/E2/E4. Voxel/Slat-Merge (coordinate-exact latent feature copy) = latent-level analog of our UV-space E1b merge. Their Nano3D-Edit-100k pairs are both-sides-synthetic (image→3D→edit); ours edit real artist assets. Also uses Qwen-Image-Edit for the 2D step — independent validation of our editor choice. |
| **VoxHammer** — *Training-Free Precise and Coherent 3D Editing in Native 3D Space* (3DV'26 Oral), arXiv:2508.19247, github.com/Nelipot-Lee/VoxHammer | Inversion + per-timestep latent & attention-KV caching with masked replacement — highest-fidelity preservation route for phase-2 E1c (needs explicit 3D region; we have P3-SAM). **Edit3D-Bench**: human-annotated edit-region benchmark + modular metric suite (PSNR/SSIM/LPIPS/DINO-I, FID, FVD, Chamfer, CLIP-T) — planned third gate for our auto-verification. |
| **FlowEdit** (Kulikov et al.) | The underlying inversion-free flow-editing algorithm Nano3D ports to 3D. |
| **EVA01** | Procedural edit-op taxonomy (rigid/topological op families) that seeded E3/E6-E9; interleaved multi-turn format for E10. |
| **Steer3D** | 2D-edit → regenerate → filter fallback route (our E1c fallback; confirmed its ceiling). |
| **UniVerse3D** | Scaled part-replacement editing data (~400K) — precedent for E4 volume targets. |
| **Native-3D-Editing** | Procedural removal as exact supervision (E3 primary route). |
| **3DEditVerse / Part-X-MLLM / PhysForge / ShapeLLM-Omni / DeepMesh** | Surveyed for taxonomy coverage & tokenizer/editing interfaces; no components consumed in V1. |

## Our own upstream (same project)

- SketchfabV1 asset corpus + VLM quality/complexity judge (440K "yes" pool; `part_complexity`
  axis gates part ops) — see `weikaih04/vlm-quality-complexity-judge`.
- Holistic + texture captions (Qwen3.6-27B): material vocabulary for E1a and the
  "restore original appearance" instructions of E1a_rev.
- Full design doc: research hub report *Editing Data Engine Plan* (V1 FINAL) +
  *12-task pilot pairs* interactive QC viewer.
