"""Pilot: TRELLIS.2 texturing pipeline on pilot assets.
Mode 'null': texture with the ORIGINAL front view (null edit) -> measures
  geometry-frozen fidelity + texture re-synthesis identity in one shot.
Mode 'edit': texture with a pre-edited image (E1a real pairs; expects
  edited images under EDITED_DIR/<sha>.png).

Usage: python pilot_texture.py <mode> <shard_idx> <n_shards> [limit]
Output: out_pilot/<mode>/<sha>/{after.glb, views_*.png, meta.json}
Resume-safe: skips shas whose meta.json exists.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys, json, time, traceback
import numpy as np
import torch
import trimesh
from PIL import Image

MODE = sys.argv[1]
SHARD = int(sys.argv[2]) if len(sys.argv) > 2 else 0
NSHARDS = int(sys.argv[3]) if len(sys.argv) > 3 else 1
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9

BASE = "/fsx/hyperpod/weikaih_edit"
EDITED_DIR = os.environ.get("EDITED_DIR", f"{BASE}/edited_views")
OUT = os.environ.get("PILOT_OUT", f"{BASE}/out_pilot/{MODE}")
os.makedirs(OUT, exist_ok=True)

assets = json.load(open(f"{BASE}/pilot200.json"))
if os.environ.get("ONLY_EDITED"):
    assets = [a for a in assets if os.path.exists(f"{EDITED_DIR}/{a['sha']}.png")]
assets = assets[SHARD::NSHARDS][:LIMIT]
print(f"[{MODE}] shard {SHARD}/{NSHARDS}: {len(assets)} assets", flush=True)

from trellis2.pipelines import Trellis2TexturingPipeline
from trellis2.utils import render_utils
import trellis2.pipelines.rembg as _rembg_mod
import o_voxel

# our render views are RGBA with real alpha -> rembg never invoked; RMBG-2.0 is
# HF-gated so stub it out instead of downloading
class _NoRembg:
    def __init__(self, **kw): pass
    def to(self, *a, **kw): return self
    def cpu(self): return self
    def __call__(self, *a, **kw):
        raise RuntimeError("rembg stubbed out — feed RGBA images with alpha")
_rembg_mod.BiRefNet = _NoRembg

pipe = Trellis2TexturingPipeline.from_pretrained(
    "microsoft/TRELLIS.2-4B", config_file="texturing_pipeline.json")
pipe.cuda()
print("pipeline loaded", flush=True)


def render_views(mesh_t2, out_prefix, n=4):
    # render n views of the TRELLIS.2 mesh output
    frames, _, _ = render_utils.render_multiview(mesh_t2, resolution=512, nviews=n)
    for i, fr in enumerate(frames):
        Image.fromarray(np.asarray(fr)).save(f"{out_prefix}_{i}.png")


for a in assets:
    sha = a["sha"]
    od = f"{OUT}/{sha}"
    if os.path.exists(f"{od}/meta.json"):
        continue
    os.makedirs(od, exist_ok=True)
    t0 = time.time()
    try:
        mesh = trimesh.load(a["glb"], force="mesh")
        if MODE == "null":
            img = Image.open(a["front_view"])
        else:
            ip = f"{EDITED_DIR}/{sha}.png"
            if not os.path.exists(ip):
                continue
            img = Image.open(ip)
        out_mesh = pipe.run(mesh, img, resolution=512, texture_size=1024)
        # texturing pipeline returns a finished textured trimesh — export directly
        try:
            out_mesh.export(f"{od}/after.glb", extension_webp=True)
        except TypeError:
            out_mesh.export(f"{od}/after.glb")
        img.save(f"{od}/cond_image.png")
        json.dump({"sha": sha, "mode": MODE, "secs": round(time.time()-t0, 1)},
                  open(f"{od}/meta.json", "w"))
        print(f"[{sha}] done {time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        traceback.print_exc()
        json.dump({"sha": sha, "mode": MODE, "error": True,
                   "exc": traceback.format_exc()[-1500:]},
                  open(f"{od}/meta.json", "w"))
print("SHARD_DONE", flush=True)
