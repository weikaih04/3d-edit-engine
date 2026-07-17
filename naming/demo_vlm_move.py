"""VLM-merge prototype, step 3: move a semantic group chosen by the VLM and
compare against the geometry-only co-move group.
Picks the first MULTI-color group per asset (where VLM merging actually differs),
falling back to the largest group. Exports before/geo/vlm glbs.
Usage: python demo_vlm_move.py
Output: vlm_move_demo/<sha>__<groupname>/{before,geo,vlm}.glb + case.json
"""
import os, json, glob
import numpy as np
from edit_parts_lib import B, pilot, load_dump, export_scene, face_slices

OUT = f"{B}/vlm_move_demo"


def transform_parts(meshes, fids, pids, T):
    out = []
    s = set(int(p) for p in pids)
    for m, (a, b) in zip(meshes, face_slices(meshes)):
        mm = m.copy()
        pf = np.where(np.isin(fids[a:b], list(s)))[0]
        if len(pf):
            vids = np.unique(mm.faces[pf])
            v = mm.vertices[vids]
            mm.vertices[vids] = (np.c_[v, np.ones(len(v))] @ T.T)[:, :3]
        out.append(mm)
    return out


for gd in sorted(glob.glob(f"{B}/vlm_merge/*/groups.json")):
    sha = gd.split('/')[-2]
    od0 = os.path.dirname(gd)
    groups = json.load(open(gd)).get('groups', [])
    colors = json.load(open(f"{od0}/colors.json"))
    tp = f"{B}/out_p3sam/all/{sha}/part_tree.json"
    tree_data = json.load(open(tp))
    gmap = {int(k): v for k, v in tree_data['group'].items()}
    multi = [g for g in groups if len(g['colors']) > 1]
    g = multi[0] if multi else max(groups, key=lambda x: len(x['colors']), default=None)
    if g is None:
        continue
    # vlm group pids: union of member colors' pids, each expanded by co-move group
    vlm_pids = set()
    for c in g['colors']:
        if c not in colors:
            continue
        for p in colors[c]['pids']:
            vlm_pids |= set(gmap.get(int(p), [int(p)]))
    if not vlm_pids:
        continue
    # geometry baseline: co-move group of the first color's first pid only
    p0 = int(colors[g['colors'][0]]['pids'][0])
    geo_pids = set(gmap.get(p0, [p0]))

    meshes, fids = load_dump(sha)
    if meshes is None:
        continue
    ext = np.ptp(np.concatenate([m.vertices for m in meshes]), axis=0)
    T = np.eye(4)
    ax = int(np.argmax(ext))
    T[:3, 3] = np.eye(3)[ax] * 0.35 * ext[ax]
    name = g['name'].replace(' ', '_')[:20]
    od = f"{OUT}/{sha}__{name}"
    os.makedirs(od, exist_ok=True)
    export_scene(meshes, f"{od}/before.glb")
    export_scene(transform_parts(meshes, fids, geo_pids, T), f"{od}/geo.glb")
    export_scene(transform_parts(meshes, fids, vlm_pids, T), f"{od}/vlm.glb")
    json.dump({"sha": sha, "group": g, "vlm_pids": sorted(vlm_pids),
               "geo_pids": sorted(geo_pids), "differs": vlm_pids != geo_pids},
              open(f"{od}/case.json", "w"), indent=1)
    print(f"[{sha[:8]}] move '{g['name']}': geo={len(geo_pids)} pids vs vlm={len(vlm_pids)} pids "
          f"{'(DIFFERS)' if vlm_pids != geo_pids else '(same)'}", flush=True)
print("VLM_MOVE_DEMO_DONE")
