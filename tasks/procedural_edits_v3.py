"""Procedural round-3: SEMANTIC-GROUP edit units.
Edit unit = VLM semantic group (colors -> clusters -> co-move pid set), named by the
VLM itself — no separate part-naming pass needed. Falls back to geometry co-move
groups for assets without VLM grouping.
Generates E3 remove / E6 transform / E7 duplicate / E10 chains into out_pairs3/.
Usage: python procedural_edits_v3.py
"""
import os, json, glob
import numpy as np
import trimesh
from edit_parts_lib import (B, pilot, load_dump, export_scene, apply_face_mask,
                            face_slices, sample_with_parts, visibility, part_gate)

OUT = f"{B}/out_pairs3"
rng = np.random.RandomState(77)
N_E3, N_E6, N_E7, N_E10 = 12, 12, 10, 8
MAX_AREA_SHARE = 0.55          # a unit covering more than this is "the whole object"
MIN_AREA_SHARE = 0.02


def semantic_units(sha):
    """[(name, pid_set, area_share)] from VLM groups expanded by co-move map."""
    gp = f"{B}/vlm_merge/{sha}/groups.json"
    cp = f"{B}/vlm_merge/{sha}/colors.json"
    tp = f"{B}/out_p3sam/all/{sha}/part_tree.json"
    if not all(os.path.exists(p) for p in [gp, cp, tp]):
        return None
    groups = json.load(open(gp)).get('groups', [])
    colors = json.load(open(cp))
    gmap = {int(k): v for k, v in json.load(open(tp))['group'].items()}
    units = []
    for g in groups:
        pids = set()
        for c in g['colors']:
            if c not in colors:
                continue
            for p in colors[c]['pids']:
                pids |= set(gmap.get(int(p), [int(p)]))
        if pids:
            units.append((g['name'].strip().lower(), pids))
    return units


def transform_pids(meshes, fids, pids, T):
    out = []
    s = list(set(int(p) for p in pids))
    for m, (a, b) in zip(meshes, face_slices(meshes)):
        mm = m.copy()
        pf = np.where(np.isin(fids[a:b], s))[0]
        if len(pf):
            vids = np.unique(mm.faces[pf])
            v = mm.vertices[vids]
            mm.vertices[vids] = (np.c_[v, np.ones(len(v))] @ T.T)[:, :3]
        out.append(mm)
    return out


def write_pair(task, name, before, after, meta):
    od = f"{OUT}/{task}/{name}"
    os.makedirs(od, exist_ok=True)
    export_scene(before, f"{od}/before.glb")
    export_scene(after, f"{od}/after.glb")
    json.dump(meta, open(f"{od}/meta.json", "w"), indent=1)
    return od


shas = [p.split('/')[-2] for p in sorted(glob.glob(f"{B}/vlm_merge/*/groups.json"))]
rng.shuffle(shas)
made = {k: 0 for k in ['E3', 'E6', 'E7', 'E10']}

for sha in shas:
    if all(made[k] >= n for k, n in
           [('E3', N_E3), ('E6', N_E6), ('E7', N_E7), ('E10', N_E10)]):
        break
    if not part_gate(sha):
        continue
    units = semantic_units(sha)
    if not units:
        continue
    meshes, fids = load_dump(sha)
    if meshes is None:
        continue
    merged_area = np.concatenate([m.area_faces for m in meshes])
    tot_area = merged_area.sum()
    cache = sample_with_parts(meshes, fids)
    pts, ppid = cache
    ext = np.ptp(np.concatenate([m.vertices for m in meshes]), axis=0)

    usable = []
    for name, pids in units:
        share = float(merged_area[np.isin(fids, list(pids))].sum() / tot_area)
        if not (MIN_AREA_SHARE <= share <= MAX_AREA_SHARE):
            continue
        # unit visibility: relabel samples (unit vs rest) and reuse the z-buffer metric
        upid = np.where(np.isin(ppid, list(pids)), 1, 0)
        vis = visibility(pts, upid, 1)
        if vis < 0.3:
            continue
        usable.append({"name": name, "pids": sorted(pids), "share": share, "vis": vis})
    if not usable:
        continue

    def pick(exclude_names=()):
        pool = [u for u in usable if u['name'] not in exclude_names]
        if not pool:
            return None
        w = np.array([u['vis'] * u['share'] for u in pool])
        return pool[int(rng.choice(len(pool), p=w / w.sum()))]

    # E3 remove
    if made['E3'] < N_E3:
        u = pick()
        if u:
            after = apply_face_mask(meshes, ~np.isin(fids, u['pids']))
            if after:
                write_pair('E3', f"{sha}__{u['name'].replace(' ', '_')}", meshes, after,
                           {"sha": sha, "task": "E3", "unit": u,
                            "instruction": f"remove the {u['name']}"})
                made['E3'] += 1

    # E6 transform
    if made['E6'] < N_E6:
        u = pick()
        if u:
            tc = np.concatenate([m.triangles_center for m in meshes])
            sel = tc[np.isin(fids, u['pids'])]
            c = sel.mean(axis=0)
            op = ['scale', 'rotate', 'translate'][made['E6'] % 3]
            T = np.eye(4)
            if op == 'scale':
                T[:3, :3] *= 1.5; T[:3, 3] = c - 1.5 * c
                instr, inv = (f"make the {u['name']} about 1.5x larger",
                              f"make the {u['name']} about a third smaller")
            elif op == 'rotate':
                T = trimesh.transformations.rotation_matrix(np.deg2rad(35), [0, 0, 1], c)
                instr, inv = (f"rotate the {u['name']} by 35 degrees",
                              f"rotate the {u['name']} back by 35 degrees")
            else:
                ax = int(np.argmax(ext[:2])); off = np.zeros(3); off[ax] = 0.3 * ext[ax]
                T[:3, 3] = off
                instr, inv = (f"shift the {u['name']} sideways",
                              f"shift the {u['name']} back to its place")
            after = transform_pids(meshes, fids, u['pids'], T)
            write_pair('E6', f"{sha}__{u['name'].replace(' ', '_')}_{op}", meshes, after,
                       {"sha": sha, "task": "E6", "op": op, "unit": u,
                        "instruction": instr, "inverse_instruction": inv})
            made['E6'] += 1

    # E7 duplicate / mirror
    if made['E7'] < N_E7:
        u = pick()
        if u and u['share'] <= 0.4:
            part = apply_face_mask(meshes, np.isin(fids, u['pids']))
            if part:
                tc = np.concatenate([m.triangles_center for m in meshes])
                c = tc[np.isin(fids, u['pids'])].mean(axis=0)
                allc = tc.mean(axis=0)
                op = 'mirror' if abs(c[0] - allc[0]) > 0.12 * ext[0] else 'duplicate'
                dup = [q.copy() for q in part]
                if op == 'mirror':
                    M = np.eye(4); M[0, 0] = -1; M[0, 3] = 2 * allc[0]
                    for q in dup:
                        q.apply_transform(M)
                    instr = f"add a mirrored copy of the {u['name']} on the other side"
                else:
                    pext = np.ptp(np.concatenate([q.vertices for q in dup]), axis=0)
                    off = np.zeros(3); ax = int(np.argmax(ext[:2]))
                    off[ax] = pext[ax] * 1.15
                    for q in dup:
                        q.apply_translation(off)
                    instr = f"add a second {u['name']} next to it"
                write_pair('E7', f"{sha}__{u['name'].replace(' ', '_')}_{op}",
                           meshes, meshes + dup,
                           {"sha": sha, "task": "E7", "op": op, "unit": u,
                            "instruction": instr})
                made['E7'] += 1

    # E10 chain: remove u1 THEN transform u2
    if made['E10'] < N_E10 and len(usable) >= 2:
        u1 = pick()
        u2 = pick(exclude_names=(u1['name'],)) if u1 else None
        if u1 and u2:
            keep = ~np.isin(fids, u1['pids'])
            s1, kept = [], []
            for m, (a, b) in zip(meshes, face_slices(meshes)):
                mm = m.copy(); k = keep[a:b]
                if not k.any():
                    continue
                if not k.all():
                    mm.update_faces(k); mm.remove_unreferenced_vertices()
                if len(mm.faces):
                    s1.append(mm); kept.append(fids[a:b][k])
            f1 = np.concatenate(kept)
            if np.isin(f1, u2['pids']).sum() >= 40:
                tc1 = np.concatenate([m.triangles_center for m in s1])
                c = tc1[np.isin(f1, u2['pids'])].mean(axis=0)
                T = np.eye(4); T[:3, :3] *= 1.5; T[:3, 3] = c - 1.5 * c
                s2 = transform_pids(s1, f1, u2['pids'], T)
                od = f"{OUT}/E10/{sha}"
                os.makedirs(od, exist_ok=True)
                export_scene(meshes, f"{od}/state0.glb")
                export_scene(s1, f"{od}/state1.glb")
                export_scene(s2, f"{od}/state2.glb")
                json.dump({"sha": sha, "task": "E10",
                           "turns": [
                               {"unit": u1, "instruction": f"remove the {u1['name']}"},
                               {"unit": u2, "instruction": f"make the {u2['name']} about 1.5x larger"}]},
                          open(f"{od}/meta.json", "w"), indent=1)
                made['E10'] += 1
    print(sha[:8], made, flush=True)

print("MADE:", made)
print("PROCEDURAL_V3_DONE")
