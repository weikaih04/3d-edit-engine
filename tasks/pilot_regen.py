"""E5 global stylization: QIE-edited view -> full TRELLIS.2 image-to-3D regeneration.
Usage: python pilot_regen.py <shard> <nshards>
Reads edited views from e5_views/<sha>.png, writes out_pilot/e5/<sha>/{after.glb, cond_image.png, meta.json}
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys, json, time, traceback, glob
import torch
from PIL import Image

SHARD = int(sys.argv[1]) if len(sys.argv) > 1 else 0
NSHARDS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
BASE = "/fsx/hyperpod/weikaih_edit"
VIEWS = f"{BASE}/e5_views"
OUT = f"{BASE}/out_pilot/e5"
os.makedirs(OUT, exist_ok=True)

shas = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{VIEWS}/*.png"))[SHARD::NSHARDS]
print(f"[e5] shard {SHARD}/{NSHARDS}: {len(shas)}", flush=True)

from trellis2.pipelines import Trellis2ImageTo3DPipeline
import trellis2.pipelines.rembg as _rembg_mod
import o_voxel

class _NoRembg:
    def __init__(self, **kw): pass
    def to(self, *a, **kw): return self
    def cpu(self): return self
    def __call__(self, *a, **kw):
        raise RuntimeError("rembg stubbed out — feed RGBA images with alpha")
_rembg_mod.BiRefNet = _NoRembg

pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipe.cuda()
print("pipeline loaded", flush=True)

for sha in shas:
    od = f"{OUT}/{sha}"
    if os.path.exists(f"{od}/meta.json"):
        continue
    os.makedirs(od, exist_ok=True)
    t0 = time.time()
    try:
        img = Image.open(f"{VIEWS}/{sha}.png")
        mesh = pipe.run(img, pipeline_type='512')[0]
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces,
            attr_volume=mesh.attrs, coords=mesh.coords,
            attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=500000, texture_size=1024,
            remesh=True, remesh_band=1, remesh_project=0, verbose=False)
        glb.export(f"{od}/after.glb", extension_webp=True)
        img.save(f"{od}/cond_image.png")
        json.dump({"sha": sha, "task": "E5", "secs": round(time.time()-t0, 1)},
                  open(f"{od}/meta.json", "w"))
        print(f"[{sha}] done {time.time()-t0:.1f}s", flush=True)
    except Exception:
        traceback.print_exc()
        json.dump({"sha": sha, "task": "E5", "error": True,
                   "exc": traceback.format_exc()[-1500:]},
                  open(f"{od}/meta.json", "w"))
print("E5_SHARD_DONE", flush=True)
