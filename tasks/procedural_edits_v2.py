"""Procedural round-2: E3/E6/E7/E9/E10 with area-share + visibility part selection
and part_complexity gating. Outputs out_pairs2/<TASK>/...
Usage: python procedural_edits_v2.py
"""
import os, json, glob
import numpy as np
import trimesh
from edit_parts_lib import (B, pilot, load_dump, part_stats_v2, export_scene,
                            apply_face_mask, transform_part_vertices,
                            save_part_ref, part_gate, face_slices)

OUT = f"{B}/out_pairs2"
rng = np.random.RandomState(42)
N_E3, N_E6, N_E7, N_E9, N_E10 = 10, 10, 8, 8, 6


def write_pair(task, name, before, after, meta):
    od = f"{OUT}/{task}/{name}"
    os.makedirs(od, exist_ok=True)
    export_scene(before, f"{od}/before.glb")
    export_scene(after, f"{od}/after.glb")
    json.dump(meta, open(f"{od}/meta.json", "w"), indent=1)
    return od


def fresh(meshes):
    """rebuild Trimesh objects so normals are recomputed after vertex edits"""
    out = []
    for m in meshes:
        out.append(trimesh.Trimesh(vertices=m.vertices.copy(), faces=m.faces.copy(),
                                   visual=m.visual, process=False))
    return out


cands = [os.path.basename(d.rstrip('/')) for d in sorted(glob.glob(f"{B}/out_p3sam/all/*/"))]
cands = [s for s in cands if s in pilot()]
rng.shuffle(cands)

made = {k: 0 for k in ['E3', 'E6', 'E7', 'E9', 'E10']}
for sha in cands:
    if all(made[k] >= n for k, n in
           [('E3', N_E3), ('E6', N_E6), ('E7', N_E7), ('E9', N_E9), ('E10', N_E10)]):
        break
    meshes, fids = load_dump(sha)
    if meshes is None:
        continue
    ext = np.ptp(np.concatenate([m.vertices for m in meshes]), axis=0)

    # --- E9 deformation: any asset, no parts needed
    if made['E9'] < N_E9:
        op = ['twist', 'taper', 'bulge'][made['E9'] % 3]
        after = fresh(meshes)
        zs = np.concatenate([m.vertices[:, 2] for m in meshes])
        z0, z1 = zs.min(), zs.max()
        allv = np.concatenate([m.vertices for m in meshes])
        cx, cy = allv[:, 0].mean(), allv[:, 1].mean()   # GLOBAL center (was per-submesh)
        for mm in after:
            v = mm.vertices.copy()
            t = (v[:, 2] - z0) / max(z1 - z0, 1e-6)
            if op == 'twist':
                ang = np.deg2rad(40) * t
                x, y = v[:, 0] - cx, v[:, 1] - cy
                v[:, 0] = cx + x * np.cos(ang) - y * np.sin(ang)
                v[:, 1] = cy + x * np.sin(ang) + y * np.cos(ang)
            elif op == 'taper':
                s = 1.0 - 0.45 * t
                v[:, 0] = cx + (v[:, 0] - cx) * s
                v[:, 1] = cy + (v[:, 1] - cy) * s
            else:
                s = 1.0 + 0.35 * np.exp(-((t - 0.5) ** 2) / 0.05)
                v[:, 0] = cx + (v[:, 0] - cx) * s
                v[:, 1] = cy + (v[:, 1] - cy) * s
            mm.vertices = v
        write_pair('E9', f"{sha}__{op}", meshes, fresh(after),
                   {"sha": sha, "task": "E9", "op": op,
                    "instruction_template": {
                        'twist': "twist the object around its vertical axis",
                        'taper': "taper the object so it narrows towards the top",
                        'bulge': "make the middle of the object bulge outwards"}[op]})
        made['E9'] += 1

    # --- part tasks: gate on judge part_complexity
    if not part_gate(sha):
        continue
    parts, pts_cache = part_stats_v2(meshes, fids)
    if not parts:
        continue

    def pick(exclude=None):
        pool = [p for p in parts if p['pid'] != exclude]
        if not pool:
            return None
        # weight towards visible & mid-size parts
        w = np.array([p['vis'] * p['area_share'] for p in pool])
        return pool[int(rng.choice(len(pool), p=w / w.sum()))]

    # --- E3 removal
    if made['E3'] < N_E3:
        p = pick()
        if p:
            after = apply_face_mask(meshes, fids != p['pid'])
            if after:
                od = write_pair('E3', f"{sha}__p{p['pid']}", meshes, after,
                                {"sha": sha, "task": "E3", "part_id": p['pid'],
                                 "area_share": p['area_share'], "vis": p['vis'],
                                 "instruction_template": "remove the {PART}"})
                save_part_ref(meshes, fids, p['pid'], sha, f"{od}/part_ref.jpg", pts_cache)
                made['E3'] += 1

    # --- E6 rigid transform
    if made['E6'] < N_E6:
        p = pick()
        if p:
            pid = p['pid']
            tc = np.concatenate([m.triangles_center for m in meshes])
            c = tc[fids == pid].mean(axis=0)
            op = ['scale', 'rotate', 'translate'][made['E6'] % 3]
            T = np.eye(4)
            if op == 'scale':
                s = 1.5
                T[:3, :3] *= s
                T[:3, 3] = c - s * c
                desc = "make the {PART} about 1.5x larger"
            elif op == 'rotate':
                T = trimesh.transformations.rotation_matrix(np.deg2rad(35), [0, 0, 1], c)
                desc = "rotate the {PART} by 35 degrees"
            else:
                off = np.zeros(3)
                ax = int(np.argmax(ext[:2]))
                off[ax] = 0.3 * ext[ax]
                T[:3, 3] = off
                desc = "shift the {PART} sideways"
            after = transform_part_vertices(meshes, fids, pid, T)
            od = write_pair('E6', f"{sha}__p{pid}_{op}", meshes, fresh(after),
                            {"sha": sha, "task": "E6", "part_id": pid, "op": op,
                             "area_share": p['area_share'], "vis": p['vis'],
                             "inverse_template": {
                                 'scale': "make the {PART} about a third smaller",
                                 'rotate': "rotate the {PART} back by 35 degrees",
                                 'translate': "shift the {PART} back to the center"}[op],
                             "instruction_template": desc})
            save_part_ref(meshes, fids, pid, sha, f"{od}/part_ref.jpg", pts_cache)
            made['E6'] += 1

    # --- E7 duplicate / mirror
    if made['E7'] < N_E7:
        p = pick()
        if p:
            pid = p['pid']
            part = apply_face_mask(meshes, fids == pid)
            if part:
                tc = np.concatenate([m.triangles_center for m in meshes])
                c = tc[fids == pid].mean(axis=0)
                allc = tc.mean(axis=0)
                op = 'mirror' if abs(c[0] - allc[0]) > 0.12 * ext[0] else 'duplicate'
                dup = [q.copy() for q in part]
                if op == 'mirror':
                    M = np.eye(4); M[0, 0] = -1; M[0, 3] = 2 * allc[0]
                    for q in dup:
                        q.apply_transform(M)
                    desc = "add a mirrored copy of the {PART} on the other side"
                else:
                    pext = np.ptp(np.concatenate([q.vertices for q in dup]), axis=0)
                    off = np.zeros(3); ax = int(np.argmax(ext[:2]))
                    off[ax] = pext[ax] * 1.15
                    for q in dup:
                        q.apply_translation(off)
                    desc = "add a second copy of the {PART} next to it"
                od = write_pair('E7', f"{sha}__p{pid}_{op}", meshes, meshes + dup,
                                {"sha": sha, "task": "E7", "part_id": pid, "op": op,
                                 "area_share": p['area_share'], "vis": p['vis'],
                                 "instruction_template": desc})
                save_part_ref(meshes, fids, pid, sha, f"{od}/part_ref.jpg", pts_cache)
                made['E7'] += 1

    # --- E10 chain: remove p1 THEN scale p2  (distinct, both visible)
    if made['E10'] < N_E10 and len(parts) >= 2:
        p1 = pick()
        p2 = pick(exclude=p1['pid']) if p1 else None
        if p1 and p2:
            keep = fids != p1['pid']
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
            if (f1 == p2['pid']).sum() >= 40:
                tc1 = np.concatenate([m.triangles_center for m in s1])
                c = tc1[f1 == p2['pid']].mean(axis=0)
                T = np.eye(4); T[:3, :3] *= 1.5; T[:3, 3] = c - 1.5 * c
                s2 = transform_part_vertices(s1, f1, p2['pid'], T)
                od = f"{OUT}/E10/{sha}"
                os.makedirs(od, exist_ok=True)
                export_scene(meshes, f"{od}/state0.glb")
                export_scene(s1, f"{od}/state1.glb")
                export_scene(fresh(s2), f"{od}/state2.glb")
                save_part_ref(meshes, fids, p1['pid'], sha, f"{od}/part1_ref.jpg", pts_cache)
                save_part_ref(meshes, fids, p2['pid'], sha, f"{od}/part2_ref.jpg", pts_cache)
                json.dump({"sha": sha, "task": "E10",
                           "turns": [
                               {"part_id": p1['pid'], "instruction_template": "remove the {PART1}"},
                               {"part_id": p2['pid'], "instruction_template": "make the {PART2} about 1.5x larger"}]},
                          open(f"{od}/meta.json", "w"), indent=1)
                made['E10'] += 1
    print(sha[:8], made, flush=True)

print("MADE:", made)
print("PROCEDURAL_V2_DONE")
