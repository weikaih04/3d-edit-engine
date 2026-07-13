"""X-Part full-chain stage 1 (p3sam env, GPU): for E4/E8 v1 pairs, re-synthesize
the crudely placed donor part in context.
  donor submeshes = trailing geometries of after.glb beyond before.glb's count
  bbox = donor bounds (x1.15) -> X-Part(mesh=after_normalized, aabb) -> gen part
  composed_geo.glb = (after minus donor) + generated part   [untextured geometry]
Usage: python xpart_chain_geo.py <task E4|E8> [n]
"""
import os, sys, json, glob
import numpy as np
import trimesh

B = "/fsx/hyperpod/weikaih_edit"
TASK = sys.argv[1] if len(sys.argv) > 1 else "E4"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
OUT = f"{B}/out_pairs2/{TASK}"
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, f"{B}/Hunyuan3D-Part/XPart")
from partgen.partformer_pipeline import PartFormerPipeline
import torch

pipeline = PartFormerPipeline.from_pretrained(model_path="tencent/Hunyuan3D-Part", verbose=False)
pipeline.to(device="cuda", dtype=torch.float32)
print("XPART_LOADED", flush=True)

done = 0
for mp in sorted(glob.glob(f"{B}/out_pairs/{TASK}/*/meta.json")):
    if done >= N:
        break
    d = os.path.dirname(mp)
    name = os.path.basename(d)
    od = f"{OUT}/{name}"
    if os.path.exists(f"{od}/composed_geo.glb"):
        done += 1
        continue
    try:
        meta = json.load(open(mp))
        bsc = trimesh.load(f"{d}/before.glb")
        asc = trimesh.load(f"{d}/after.glb")
        bg = list(bsc.geometry.values()) if isinstance(bsc, trimesh.Scene) else [bsc]
        ag = list(asc.geometry.values()) if isinstance(asc, trimesh.Scene) else [asc]
        if len(ag) <= len(bg):
            print(f"[{name[:24]}] no trailing donor, skip", flush=True)
            continue
        donor = ag[len(bg):]
        base = ag[:len(bg)]
        dv = np.concatenate([g.vertices for g in donor])
        lo, hi = dv.min(0), dv.max(0)
        c, ext = (lo + hi) / 2, hi - lo
        lo, hi = c - ext * 0.575, c + ext * 0.575          # x1.15 box

        merged = trimesh.util.concatenate([g.copy() for g in ag])
        v = merged.vertices
        mlo, mhi = v.min(0), v.max(0)
        center = (mlo + mhi) / 2
        scale = (mhi - mlo).max() / 2 / 0.8
        merged.vertices = (v - center) / scale
        aabb = np.array([[(lo - center) / scale, (hi - center) / scale]], dtype=np.float32)

        out, _ = pipeline(mesh=merged, aabb=aabb, octree_resolution=384, output_type="trimesh")
        gen = out if isinstance(out, trimesh.Trimesh) else \
            trimesh.util.concatenate([g for g in out.geometry.values()])
        gen.vertices = gen.vertices * scale + center

        os.makedirs(od, exist_ok=True)
        comp = trimesh.Scene()
        for i, g in enumerate(base):
            comp.add_geometry(g, geom_name=f"g{i}")
        comp.add_geometry(gen, geom_name="gen_part")
        comp.export(f"{od}/composed_geo.glb")
        os.link(f"{d}/before.glb", f"{od}/before.glb") if not os.path.exists(f"{od}/before.glb") else None
        if os.path.exists(f"{d}/part_ref.jpg") and not os.path.exists(f"{od}/part_ref.jpg"):
            os.link(f"{d}/part_ref.jpg", f"{od}/part_ref.jpg")
        json.dump({**meta, "source": "xpart_chain", "gen_faces": int(len(gen.faces))},
                  open(f"{od}/meta.json", "w"), indent=1)
        done += 1
        print(f"[{name[:24]}] gen_faces={len(gen.faces)}", flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
print(f"XPART_CHAIN_GEO_DONE {TASK} {done}", flush=True)
