"""E9 v2.1: regenerate deformation pairs on CLEAN assets only (few submeshes,
no multi-shell scans that tear) with stronger ops. Replaces out_pairs2/E9.
Usage: python e9_v21.py
"""
import os, json, glob, shutil
import numpy as np
import trimesh
from edit_parts_lib import B, pilot, export_scene

OUT = f"{B}/out_pairs2/E9"
N = 8
rng = np.random.RandomState(5)

cands = []
for sha, a in pilot().items():
    try:
        sc = trimesh.load(a['glb'])
        meshes = sc.dump() if isinstance(sc, trimesh.Scene) else [sc]
        meshes = [m for m in meshes if isinstance(m, trimesh.Trimesh) and len(m.faces)]
    except Exception:
        continue
    nf = sum(len(m.faces) for m in meshes)
    if not meshes or len(meshes) > 8 or nf > 200000 or nf < 800:
        continue
    if a.get('category') in ('scan', 'terrain'):
        continue
    cands.append((sha, meshes))
    if len(cands) >= 40:
        break
rng.shuffle(cands)
print(f"{len(cands)} clean candidates")

shutil.rmtree(OUT, ignore_errors=True)
made = 0
for sha, meshes in cands:
    if made >= N:
        break
    op = ['twist', 'taper', 'bulge', 'bend'][made % 4]
    allv = np.concatenate([m.vertices for m in meshes])
    z0, z1 = allv[:, 2].min(), allv[:, 2].max()
    cx, cy = allv[:, 0].mean(), allv[:, 1].mean()
    ext = np.ptp(allv, axis=0)
    after = []
    for m in meshes:
        mm = trimesh.Trimesh(vertices=m.vertices.copy(), faces=m.faces.copy(),
                             visual=m.visual, process=False)
        v = mm.vertices.copy()
        t = (v[:, 2] - z0) / max(z1 - z0, 1e-6)
        if op == 'twist':
            ang = np.deg2rad(60) * t
            x, y = v[:, 0] - cx, v[:, 1] - cy
            v[:, 0] = cx + x * np.cos(ang) - y * np.sin(ang)
            v[:, 1] = cy + x * np.sin(ang) + y * np.cos(ang)
        elif op == 'taper':
            s = 1.0 - 0.55 * t
            v[:, 0] = cx + (v[:, 0] - cx) * s
            v[:, 1] = cy + (v[:, 1] - cy) * s
        elif op == 'bulge':
            s = 1.0 + 0.5 * np.exp(-((t - 0.5) ** 2) / 0.05)
            v[:, 0] = cx + (v[:, 0] - cx) * s
            v[:, 1] = cy + (v[:, 1] - cy) * s
        else:  # bend: lean the top sideways along x
            v[:, 0] = v[:, 0] + 0.35 * ext[0] * t * t
        mm.vertices = v
        after.append(mm)
    od = f"{OUT}/{sha}__{op}"
    os.makedirs(od, exist_ok=True)
    export_scene(meshes, f"{od}/before.glb")
    export_scene(after, f"{od}/after.glb")
    json.dump({"sha": sha, "task": "E9", "op": op,
               "instruction_template": {
                   'twist': "twist the object around its vertical axis",
                   'taper': "taper the object so it narrows towards the top",
                   'bulge': "make the middle of the object bulge outwards",
                   'bend': "bend the object so it leans to the side"}[op],
               "instruction": {
                   'twist': "twist the object around its vertical axis",
                   'taper': "taper the object so it narrows towards the top",
                   'bulge': "make the middle of the object bulge outwards",
                   'bend': "bend the object so it leans to the side"}[op]},
              open(f"{od}/meta.json", "w"), indent=1)
    made += 1
    print(sha[:8], op, flush=True)
print(f"E9_V21_DONE {made}")
