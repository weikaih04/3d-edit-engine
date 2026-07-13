# Required edits to third-party checkouts

## TRELLIS.2 (github.com/microsoft/TRELLIS.2, local clone)

`trellis2/modules/image_feature_extractor.py` — transformers >= 5 moved DINOv3 layers:
```python
_layers = self.model.layer if hasattr(self.model, 'layer') else self.model.model.layer
```
(replaces the direct `self.model.layer` access; keeps compat with transformers 4.x.)

## Hunyuan3D-Part (tencent; fetched via GitHub API tarball — git clone 403s)

### P3-SAM/model.py (line ~18)
sonata checkpoint download root is hardcoded to `/root/sonata`:
```python
download_root='<your_workdir>/sonata_cache'
```

### P3-SAM/demo/auto_mask.py (`__main__` directory loop)
Stock script discards results when `--save_mid_res 0` and dies on the first bad mesh. Patch:
- resume-skip an asset if `aabb.npy` or `error.txt` exists;
- wrap each asset in try/except, writing `error.txt` on failure (numba zero-size crash on
  degenerate meshes);
- explicitly save `aabb.npy`, `face_ids.npy`, `mesh_clean.glb` (returns were dropped).

### X-Part runtime
- Model dir resolution: `export HY3DGEN_MODELS=<dir>` with
  `<dir>/tencent/Hunyuan3D-Part -> <hf_cache_snapshot>` symlink (it does NOT read the HF cache).
- Extra deps on top of the p3sam env: `pytorch_lightning scikit-image fpsample
  pymeshlab==2023.12.post3 addict torch_cluster` (pyg wheel matching torch+cu).

## Environment pins that matter
- spconv-cu121==2.3.8 (sm_90/H200; the cu120 2.3.6 wheel SIGFPEs).
- triton>=3.2 for FlexGEMM; build all TRELLIS.2 extensions against conda
  `nvidia/label/cuda-12.1.1` toolkit; `pip install --no-deps` for extension deps.
- diffusers>=0.36 for `QwenImageEditPlusPipeline` (we run 0.39).
- vLLM 0.23: `StructuredOutputsParams` for guided JSON.
