"""X-Part seam-fix validation: take E4 v1 pairs (donor crudely fitted into the
removed part's bbox), ask X-Part to RE-SYNTHESIZE that bbox region conditioned
on the whole object -> context-coherent part geometry.
Usage: python xpart_test.py [n_cases]
Output: xpart_test_out/<name>/{gen_part.glb, composed.glb, viz.jpg}
"""
import os, sys, json, glob
import numpy as np
import trimesh

B = "/fsx/hyperpod/weikaih_edit"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
OUT = f"{B}/xpart_test_out"
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, f"{B}/Hunyuan3D-Part/XPart")
from partgen.partformer_pipeline import PartFormerPipeline
import torch

pipeline = PartFormerPipeline.from_pretrained(model_path="tencent/Hunyuan3D-Part", verbose=True)
pipeline.to(device="cuda", dtype=torch.float32)
print("XPART_LOADED", flush=True)


def xpart_normalize(mesh):
    v = mesh.vertices
    lo, hi = v.min(0), v.max(0)
    center = (lo + hi) / 2
    scale = (hi - lo).max() / 2 / 0.8
    m = mesh.copy()
    m.vertices = (v - center) / scale
    return m, center, scale


cases = sorted(glob.glob(f"{B}/out_pairs/E4/*/meta.json"))[:N]
pilot = {a['sha']: a for a in json.load(open(f'{B}/pilot200.json'))}

for mp in cases:
    d = os.path.dirname(mp)
    meta = json.load(open(mp))
    sha, pid = meta['sha'], meta['part_id']
    name = os.path.basename(d)
    od = f"{OUT}/{name}"
    os.makedirs(od, exist_ok=True)
    try:
        # bbox of the replaced region = original part bounds (donor was fitted there)
        orig = trimesh.load(pilot[sha]['glb'], force='mesh')
        fids = np.load(f"{B}/out_p3sam/all/{sha}/face_ids.npy")
        tc = orig.triangles_center
        sel = tc[fids == pid]
        lo, hi = sel.min(0), sel.max(0)
        pad = 0.06 * (hi - lo).max()
        lo, hi = lo - pad, hi + pad

        after = trimesh.load(f"{d}/after.glb", force='mesh')
        norm, center, scale = xpart_normalize(after)
        aabb = np.array([[(lo - center) / scale, (hi - center) / scale]], dtype=np.float32)

        out, (out_bbox, mesh_bbox, explode) = pipeline(
            mesh=norm, aabb=aabb, octree_resolution=384,
            output_type="trimesh")
        gen = out if isinstance(out, trimesh.Trimesh) else \
            trimesh.util.concatenate([g for g in out.geometry.values()])
        gen.vertices = gen.vertices * scale + center      # back to world frame
        gen.export(f"{od}/gen_part.glb")

        # composed: original minus part + generated part
        keep = apply = None
        rest_faces = fids != pid
        rest = orig.copy()
        rest.update_faces(rest_faces)
        rest.remove_unreferenced_vertices()
        comp = trimesh.Scene()
        comp.add_geometry(rest, geom_name="rest")
        comp.add_geometry(gen, geom_name="gen_part")
        comp.export(f"{od}/composed.glb")
        json.dump({"sha": sha, "part_id": pid, "gen_faces": int(len(gen.faces))},
                  open(f"{od}/meta.json", "w"))
        print(f"[{name[:30]}] gen_part faces={len(gen.faces)}", flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
        open(f"{od}/error.txt", "w").write(__import__('traceback').format_exc())
print("XPART_TEST_DONE", flush=True)
