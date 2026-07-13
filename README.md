# 3D Edit Engine — instruction-editing pair factory for 3D assets

Generates **(before-asset, instruction, after-asset)** training pairs across a 12-type edit
taxonomy, using only **frozen off-the-shelf models** — no training anywhere in the engine:

| Role | Model / tool |
|---|---|
| 3D generation (texture-given-shape, full image-to-3D) | **TRELLIS.2-4B** (microsoft, frozen) |
| 2D instruction-driven image editing | **Qwen-Image-Edit-2511** |
| Native 3D part segmentation | **P3-SAM** (tencent/Hunyuan3D-Part) |
| In-context part re-synthesis | **X-Part** (tencent/Hunyuan3D-Part) |
| Part naming + dual-blind verification + captions | **Qwen3.6-27B** (vLLM) |
| Exact geometric edits | plain `trimesh` code (no model) |

Live QC viewer (every pair, verdict, 2D strips, drag-to-rotate GLBs):
`https://dashboard.ai-research-wk.com/reports/3d-generation-editing/editing-pairs-viewer`

## Task matrix & pilot results (dual-blind judged)

| Task | Method | Pilot pass |
|---|---|---|
| E1a global material | freeze geometry + re-sample TRELLIS.2 texture flow on QIE-edited view | 29/30 |
| E1a_rev restore | free inverse of E1a; instruction from texture captions | 25/30 |
| E1b local material | **targeted** QIE ("change only the X") + UV-space mask merge | 9/11 |
| E1c pattern-keep recolor | *deferred to phase-2 FlowEdit* (fallback ceiling 2/10) | — |
| E2 add part | exact inverse of E3 removal | 6/10 |
| E3 remove part | procedural face deletion (area-share + visibility part pick) | 9/10 |
| E4 replace part | procedural swap → X-Part re-synthesis → TRELLIS.2 re-texture | 4/6 |
| E5 stylize | QIE-edited view → full TRELLIS.2 image-to-3D regen | 8/8 |
| E6 move/scale part | procedural affine on part vertices | 17/20 |
| E7 duplicate/mirror | procedural copy/mirror | 7/8 |
| E8 cross-asset part | procedural donor paste **keeping donor texture** (X-Part chain rejected — breaks edit locality) | 3/8+5M |
| E9 deform | twist/taper/bulge/bend on clean assets | 4/8 |
| E10 multi-turn | composed ONLY from gate-passed single edits | 2+3p/5 |

## Pipeline anatomy

```
selection/   pick assets (VLM-judge axes) + parts (P3-SAM ids -> area-share + z-buffer visibility)
tasks/       per-task pair generators (procedural exact ops / QIE+TRELLIS generative / X-Part chain)
naming/      Qwen3.6-27B names highlighted parts -> fills instruction templates
verify/      auto-gate: geometry checks (silhouette IoU, visibility) + TRUE dual-blind judge
render/      blender CPU renders (bbox-anchored fair framing, look-at-part closeups), QC grids
viewer/      builds the public model-viewer QC site
engine/      shared libs: part selection (edit_parts_lib), UV-space local texture merge (e1b_merge)
slurm/       single-node 8xH200 job scripts for every stage
configs/     instruction sets + example artifacts (part names, human verdicts)
```

### Key verified invariants (the engine's load-bearing facts)

1. **P3-SAM `face_ids` map 1:1 onto trimesh scene-dump merged face order** (bitwise verified)
   — part masks apply directly to the textured original GLB.
2. **Frozen geometry ⇒ TRELLIS.2 remesh + UV atlas is deterministic**: texturing the same mesh
   twice yields bitwise-identical topology/UV ⇒ local material edits are a pure CPU
   UV-space texture merge (no extra GPU sampling). Silhouette IoU original-vs-output ≈ 0.99.
3. **TRELLIS.2 outputs live in the normalized [-0.5,0.5] cube** — any mask/attribute transfer
   from the original mesh must normalize into that frame first.
4. **Edit locality principle**: whole-object re-texture is right when the edit *should* look
   native (E4 replace), wrong when the edit must stay visually distinct (E8 cross-asset).

## Quickstart (pilot-scale)

```bash
# paths: scripts use B=/fsx/hyperpod/weikaih_edit as the data workdir; adjust the constant
export PYTHONPATH=$REPO/engine:$PYTHONPATH

# 0. select assets + segment parts (P3-SAM, see slurm/)
python selection/select_pilot_assets.py
# 1. procedural tasks (CPU)
python tasks/procedural_edits_v2.py && python tasks/reverse_pairs.py
# 2. generative tasks (GPU): QIE views -> texturing / full regen
sbatch slurm/sbatch_batch2.sh
# 3. name parts, verify, render, judge
sbatch slurm/sbatch_round2.sh
# 4. build & publish the QC viewer
python viewer/build_site.py
```

Environments: `trellis2` (torch 2.5.1+cu121 + TRELLIS.2 extensions + diffusers>=0.36),
`p3sam` (torch 2.4+cu121 + sonata/spconv-cu121==2.3.8 + torch_cluster + pytorch_lightning),
`vlmcap` (vLLM 0.23). Model caches must be local (compute nodes are offline) — see docs/GOTCHAS.md.

## Docs

- `docs/RESULTS.md` — per-round quality progression and what fixed what
- `docs/GOTCHAS.md` — every trap we hit (coordinate frames, HF offline, vLLM spawn, …)
- `docs/THIRD_PARTY_PATCHES.md` — required edits to TRELLIS.2 / Hunyuan3D-Part checkouts
