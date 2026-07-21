# Gotchas — every trap this engine hit, in one place

## Coordinate frames
- **TRELLIS.2 outputs are normalized to the [-0.5, 0.5] cube.** Any transfer from the original
  world-coordinate mesh (part masks, nearest-face lookup) must first normalize the original with
  `center=(lo+hi)/2, scale=max_extent`. Symptom of forgetting: KDTree transfer silently matches
  garbage — small objects "work", large ones fail (our frog-eye-mask-on-the-base-plate bug).
- Silhouette-IoU comparisons between original and TRELLIS output must be similarity-invariant:
  normalize EACH mesh to its own bbox before rasterizing.
- `trimesh.sample.sample_surface` is random: seed with `np.random.seed(0)` before EACH call or
  identical meshes won't produce identical occupancy grids.

## Rendering
- `render_cond.py` fov is in **radians**: `fov = 2*asin(sqrt(3)/2/radius)`.
- Blender frames each glb to its own bounds → before/after renders get different framing when
  geometry changes. Fix: append two invisible micro-triangles at the PAIR-UNION bbox corners
  ("anchored" renders). Same trick with a part-bbox gives **look-at-part closeups** (E7 5/8→7/8).
- renders_cond view ordering is elevation LOW→HIGH: view000 is a from-below shot. Pick views
  005–011 (sha-deterministic) — conditioning QIE/TRELLIS on view000 causes texture hallucination
  on everything the camera can't see.
- Blender 4.5.1 lives in node-local /tmp; compute nodes are offline — untar from an fsx tarball
  in the job prolog. Needs `LD_LIBRARY_PATH=<...>/blender_libs` (libXfixes shim).

## HF / offline compute nodes
- Compute nodes have no internet: every model must be pre-cached; set `HF_HOME` + `HF_HUB_OFFLINE=1`.
- TRELLIS.2's full image-to-3D `pipeline.json` references **microsoft/TRELLIS-image-large**
  (TRELLIS-1!) for the sparse-structure decoder — cache that too or E5 dies only-at-runtime.
- Hunyuan3D-Part loaders don't use the HF cache: they want `$HY3DGEN_MODELS/tencent/Hunyuan3D-Part`
  as a plain dir — symlink the HF snapshot there.
- DINOv3 and RMBG-2.0 are gated: graft the camenduru DINOv3 mirror into the facebook cache path;
  stub rembg (`BiRefNet`) entirely when inputs are RGBA renders.

## vLLM (Qwen3.6-27B)
- Qwen3.6 is a thinking model — disable it (`enable_thinking: False` via chat, or `/no_think`),
  else the token budget burns on reasoning and JSON never appears.
- vLLM v1 spawns the engine core with `spawn`: your script is re-imported → everything must live
  under `if __name__ == "__main__":` or the child re-executes module-level model loads.
- For machine-readable outputs use structured outputs. vLLM 0.23 renamed the API:
  `StructuredOutputsParams` (not `GuidedDecodingParams`).
- **Single-prompt edit judging is instruction-biased**: showing the instruction alongside the
  images made the judge confirm 8/14 known-bad pairs. TRUE dual-blind (stage A: describe change
  from images only; stage B: text-only match vs instruction) fixes it.

## Part operations
- Select parts by **surface-area share** (3–35%) not face-count share — low-poly parts break
  face-count logic (a candy-cane shaft is 2 faces but half the object).
- Gate part ops on VLM-judge `part_complexity >= 4`: P3-SAM on single-piece scans yields
  arbitrary regions; deleting them looks like damage, not editing.
- Add a **z-buffer visibility** check (6 axis views, frontmost-point occupancy): editing an
  invisible part produces a pair whose instruction can never be verified.
- Part-naming refs must show only VISIBLE surface points (z-buffer front layer). Raw transparent
  point scatters made Qwen name a skull's jaw "backrest". Category-consistency check
  (does this part name fit the asset's caption category?) is the next belt.

## Python / infra
- `dict.get(k, meta['x'])` evaluates the default EAGERLY — `meta['x']` raises even when `k`
  exists. Use `meta.get(k) or meta.get('x', default)`. This single pattern bit three scripts.
- P3-SAM on H200 (sm_90): spconv-cu121==2.3.8 works; spconv-cu120 2.3.6 SIGFPEs.
- FlexGEMM/nvdiffrast/etc extension builds: pin conda CUDA via `-c nvidia/label/cuda-12.1.1`
  and install python deps with `--no-deps` to stop pip replacing torch.
- pkill/pgrep patterns can match their own invoking shell (self-kill); slurm jobs read sbatch
  scripts at submit time but python files at RUN time (mid-flight edits do land).
- `hf download` replaced `huggingface-cli download`; unauthenticated downloads drop mid-way —
  wrap in retry loops; watch disk (ENOSPC on 54GB QIE) and redirect HF_HOME to the big volume.

## Texture fidelity (text/logo alignment)
- Generative texture re-synthesis (TRELLIS null cond) CANNOT reproduce text/logos —
  a 512^3 voxel color field has no room for glyph-scale high frequency; numbers come
  out as smeared blobs, livery stripes drift. So: **the unedited region of any local
  edit must NOT be re-synthesized.** Use `rebake_texture.py` — deterministic per-texel
  color transfer from the ORIGINAL asset to the TRELLIS-topology atlas (4M-point KDTree
  in the shared normalized frame, no model). E1a_rev's "after" = the original glb itself.
- Rebake caveat: thin double walls can grab the wrong side's color (nearest neighbor
  crosses the shell). Fix = filter candidates by normal agreement (dot>0) — TODO for prod.
- E1a GLOBAL material still shows garbled text: the whole surface is legitimately edited,
  so rebake-copy would undo the edit. Right fix = DETAIL TRANSFER (generative material +
  original high-freq structure layer via rebake), or OCR-tag text-heavy assets and phrase
  the instruction to "repaint over all markings" so vanishing text is intended, not a bug.

## Local material must use SEMANTIC units too
- E1b picking a single P3-SAM part paints only one shell of a multi-shell semantic part
  ("head" gold on the outer skull, eye-socket interior still original). Use the VLM
  semantic group as the mask (e1b_v3_prep.py -> e1b_merge.py pid_set), and the group NAME
  as the instruction ("change only the <group> to X"): paint-range, instruction wording,
  and 3D mask all align by construction. E1b 2/11 -> 8-9/11.

## Part removal: interior exposure ("looks like a hole")
- Removing a part rarely creates a TRUE topological hole (assets are mostly separate
  interpenetrating shells; measured new-open-boundary-edges at the cut ~0-26, minor). BUT
  it often EXPOSES interior geometry (inner walls, internal structure/cables) that reads as
  a hole/gore — the real defect for training data. ~17% of E3 severe, ~50% mild in pilot.
- Detect: fraction of after-faces near the removed region whose NORMAL points toward the
  removed part's former centroid (inner wall now facing out). `interior_exposure > 0.35`
  in geometry_checks._clean_cut → geo fail. Gate, don't fill (capping the cut is future work
  if E3 volume runs short).
- Counting total open-boundary-edge change is WRONG for hole detection: removing an open
  shell decreases the total even while creating a new cut loop. Must compare after-boundary
  vs before-boundary POSITIONS (new edges not near old ones).
