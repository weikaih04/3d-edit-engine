"""Demo: single-part op (OLD, horror) vs subtree op (NEW, fixed).
Finds assets whose part tree has a parent with 'inside' children (eyes-in-head cases),
applies the SAME sideways translation two ways, exports before/old/new glbs.
Usage: python demo_subtree_fix.py [n_demos]
Output: subtree_demo/<sha>__p<pid>/{before,old,new}.glb + case.json
"""
import os, sys, json, glob
import numpy as np
from edit_parts_lib import B, pilot, load_dump, export_scene, transform_part_vertices, face_slices

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
OUT = f"{B}/subtree_demo"


def transform_parts(meshes, fids, pids, T):
    """transform vertices touched by ANY of the given part ids"""
    import trimesh
    out = []
    pidset = set(pids)
    for m, (a, b) in zip(meshes, face_slices(meshes)):
        mm = m.copy()
        pf = np.where(np.isin(fids[a:b], list(pidset)))[0]
        if len(pf):
            vids = np.unique(mm.faces[pf])
            v = mm.vertices[vids]
            mm.vertices[vids] = (np.c_[v, np.ones(len(v))] @ T.T)[:, :3]
        out.append(mm)
    return out


cases = []
for tp in sorted(glob.glob(f"{B}/out_p3sam/all/*/part_tree.json")):
    sha = tp.split('/')[-2]
    if sha not in pilot() or pilot()[sha].get('category') not in ('character', 'creature', 'animal'):
        continue
    data = json.load(open(tp))
    if 'tree' not in data:
        continue
    tree = {int(k): v for k, v in data['tree'].items()}
    group = {int(k): v for k, v in data['group'].items()}
    parents = {}
    for c, v in tree.items():
        if v['rule'] == 'inside':
            parents.setdefault(v['parent'], []).append(c)
    if not parents:
        continue
    p = max(parents, key=lambda x: len(parents[x]))
    cases.append((sha, p, len(parents[p]), group))
cases.sort(key=lambda x: -x[2])
print(f"{len(cases)} candidate assets with inside-children")

made = 0
for sha, pid, nch, group in cases:
    if made >= N:
        break
    meshes, fids = load_dump(sha)
    if meshes is None:
        continue
    ext = np.ptp(np.concatenate([m.vertices for m in meshes]), axis=0)
    T = np.eye(4)
    T[:3, 3] = [0.35 * ext[0], 0, 0.10 * ext[2]]     # shift sideways + slightly up
    sub = group[pid]                                  # cluster + descendants co-move set
    old = transform_parts(meshes, fids, [pid], T)
    new = transform_parts(meshes, fids, sub, T)
    od = f"{OUT}/{sha}__p{pid}"
    os.makedirs(od, exist_ok=True)
    export_scene(meshes, f"{od}/before.glb")
    export_scene(old, f"{od}/old.glb")
    export_scene(new, f"{od}/new.glb")
    json.dump({"sha": sha, "parent": pid, "subtree": sub, "n_inside_children": nch},
              open(f"{od}/case.json", "w"), indent=1)
    made += 1
    print(f"[{sha[:8]}] parent p{pid}, subtree={len(sub)} parts ({nch} inside children)", flush=True)
print("DEMO_BUILD_DONE")
