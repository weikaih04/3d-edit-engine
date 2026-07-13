"""X-Part full-chain stage 2 (trellis2 env, GPU): texture the composed geometry
with TRELLIS.2 texturing conditioned on the ORIGINAL front view (null-style) —
unchanged region regains original look, the generated part gets contextual texture.
Usage: python xpart_chain_tex.py <task E4|E8> <shard> <nshards>
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys, json, time, glob, traceback
import trimesh
from PIL import Image

TASK = sys.argv[1]
SHARD = int(sys.argv[2]) if len(sys.argv) > 2 else 0
NSHARDS = int(sys.argv[3]) if len(sys.argv) > 3 else 1
B = "/fsx/hyperpod/weikaih_edit"
pilot = {a['sha']: a for a in json.load(open(f"{B}/pilot200.json"))}

from trellis2.pipelines import Trellis2TexturingPipeline
import trellis2.pipelines.rembg as _rembg_mod
class _NoRembg:
    def __init__(self, **kw): pass
    def to(self, *a, **kw): return self
    def cpu(self): return self
    def __call__(self, *a, **kw):
        raise RuntimeError("rembg stubbed")
_rembg_mod.BiRefNet = _NoRembg

pipe = Trellis2TexturingPipeline.from_pretrained(
    "microsoft/TRELLIS.2-4B", config_file="texturing_pipeline.json")
pipe.cuda()
print("pipeline loaded", flush=True)

dirs = sorted(glob.glob(f"{B}/out_pairs2/{TASK}/*/composed_geo.glb"))[SHARD::NSHARDS]
for cg in dirs:
    d = os.path.dirname(cg)
    if os.path.exists(f"{d}/after.glb"):
        continue
    t0 = time.time()
    try:
        meta = json.load(open(f"{d}/meta.json"))
        mesh = trimesh.load(cg, force='mesh')
        img = Image.open(pilot[meta['sha']]['front_view'])
        out_mesh = pipe.run(mesh, img, resolution=512, texture_size=1024)
        try:
            out_mesh.export(f"{d}/after.glb", extension_webp=True)
        except TypeError:
            out_mesh.export(f"{d}/after.glb")
        print(f"[{os.path.basename(d)[:24]}] {time.time()-t0:.1f}s", flush=True)
    except Exception:
        traceback.print_exc()
print("XPART_CHAIN_TEX_DONE", flush=True)
