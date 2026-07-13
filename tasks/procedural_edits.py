"""Procedural edit-pair generator: E3 remove / E6 rigid transform / E7 duplicate-mirror /
E9 smooth deformation / E8 cross-asset composition.
Texture-preserving: operates on the dumped (world-space) per-node textured meshes of the
original glb; P3-SAM face_ids map 1:1 onto the merged face order (verified bitwise).
Outputs out_pairs/<TASK>/<id>/{before.glb, after.glb, meta.json, part_ref.jpg}
part_ref.jpg = ortho scatter with the edited part highlighted (for VLM part naming later).
Usage: python procedural_edits.py
"""
import os, json, glob
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

B = "/fsx/hyperpod/weikaih_edit"
OUT = f"{B}/out_pairs"
rng = np.random.RandomState(0)
pilot = {a['sha']: a for a in json.load(open(f'{B}/pilot200.json'))}

N_E3, N_E6, N_E7, N_E9, N_E8 = 10, 10, 8, 8, 8


def load_dump(sha):
    """Original glb -> list of world-space textured meshes, + per-merged-face
    (mesh_idx, local_face) mapping aligned with P3-SAM face order."""
    sc = trimesh.load(pilot[sha]['glb'])
    if isinstance(sc, trimesh.Scene):
        meshes = sc.dump()
    else:
        meshes = [sc]
    meshes = [m for m in meshes if isinstance(m, trimesh.Trimesh) and len(m.faces)]
    d = f"{B}/out_p3sam/all/{sha}/"
    fids = np.load(d + 'face_ids.npy')
    nf = sum(len(m.faces) for m in meshes)
    if nf != len(fids):
        return None, None, None
    return meshes, fids, np.load(d + 'aabb.npy')


def face_slices(meshes):
    out, off = [], 0
    for m in meshes:
        out.append((off, off + len(m.faces)))
        off += len(m.faces)
    return out


def export_scene(meshes, path):
    sc = trimesh.Scene()
    for i, m in enumerate(meshes):
        sc.add_geometry(m, node_name=f"n{i}", geom_name=f"g{i}")
    sc.export(path)


def part_stats(meshes, fids):
    """usable parts: 3%-35% of faces, not the lowest (base) part"""
    uniq, cnt = np.unique(fids, return_counts=True)
    tc = np.concatenate([m.triangles_center for m in meshes])
    zmin = {u: tc[fids == u][:, 2].min() for u in uniq}
    base = min(zmin, key=zmin.get)
    share = cnt / cnt.sum()
    ok = [(u, c) for u, c, s in zip(uniq, cnt, share)
          if 0.03 <= s <= 0.35 and u != base and c > 60]
    return sorted(ok, key=lambda x: -x[1])


def save_part_ref(meshes, fids, pid, sha, path):
    merged = trimesh.util.concatenate([m.copy() for m in meshes])
    pts, fi = trimesh.sample.sample_surface(merged, 20000)
    hot = fids[fi] == pid
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    axes[0].imshow(Image.open(pilot[sha]['front_view']).convert('RGB'))
    axes[0].set_title('asset'); axes[0].axis('off')
    for ax, (i, j), nm in zip(axes[1:], [(0, 2), (1, 2), (0, 1)], ['front', 'side', 'top']):
        ax.scatter(pts[~hot][:, i], pts[~hot][:, j], c='lightgray', s=0.4)
        ax.scatter(pts[hot][:, i], pts[hot][:, j], c='red', s=0.6)
        ax.set_aspect('equal'); ax.axis('off'); ax.set_title(nm)
    plt.tight_layout(); plt.savefig(path, dpi=100); plt.close()


def apply_face_mask(meshes, keep_mask):
    """keep_mask over merged faces -> new mesh list with faces removed"""
    out = []
    for m, (a, b) in zip(meshes, face_slices(meshes)):
        mm = m.copy()
        k = keep_mask[a:b]
        if k.all():
            out.append(mm); continue
        if not k.any():
            continue
        mm.update_faces(k)
        mm.remove_unreferenced_vertices()
        if len(mm.faces):
            out.append(mm)
    return out


def extract_part(meshes, fids, pid):
    return apply_face_mask(meshes, fids == pid)


def transform_part_vertices(meshes, fids, pid, T):
    """apply 4x4 T to vertices touched by part faces (world space)"""
    out = []
    for m, (a, b) in zip(meshes, face_slices(meshes)):
        mm = m.copy()
        pf = np.where(fids[a:b] == pid)[0]
        if len(pf):
            vids = np.unique(mm.faces[pf])
            v = mm.vertices[vids]
            vh = np.c_[v, np.ones(len(v))] @ T.T
            mm.vertices[vids] = vh[:, :3]
        out.append(mm)
    return out


def write_pair(task, pid_dir, before, after, meta):
    od = f"{OUT}/{task}/{pid_dir}"
    os.makedirs(od, exist_ok=True)
    export_scene(before, f"{od}/before.glb")
    export_scene(after, f"{od}/after.glb")
    json.dump(meta, open(f"{od}/meta.json", "w"), indent=1)
    return od


# ---- asset pool: p3sam done + parts-rich first
cands = []
for d in sorted(glob.glob(f"{B}/out_p3sam/all/*/")):
    sha = os.path.basename(d.rstrip('/'))
    if sha in pilot and os.path.exists(d + 'face_ids.npy'):
        cands.append(sha)
rng.shuffle(cands)

made = {k: 0 for k in ['E3', 'E6', 'E7', 'E9', 'E8']}
loaded = {}
def get(sha):
    if sha not in loaded:
        loaded[sha] = load_dump(sha)
    return loaded[sha]

for sha in cands:
    if all(made[k] >= n for k, n in
           [('E3', N_E3), ('E6', N_E6), ('E7', N_E7), ('E9', N_E9)]):
        break
    meshes, fids, aabb = get(sha)
    if meshes is None:
        continue
    parts = part_stats(meshes, fids)
    ext = np.ptp(np.concatenate([m.vertices for m in meshes]), axis=0)

    # --- E9 deformation (no parts needed)
    if made['E9'] < N_E9:
        op = ['twist', 'taper', 'bulge'][made['E9'] % 3]
        after = [m.copy() for m in meshes]
        zs = np.concatenate([m.vertices[:, 2] for m in meshes])
        z0, z1 = zs.min(), zs.max()
        for mm in after:
            v = mm.vertices.copy()
            t = (v[:, 2] - z0) / max(z1 - z0, 1e-6)
            cx, cy = v[:, 0].mean(), v[:, 1].mean()
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
        write_pair('E9', f"{sha}__{op}", meshes, after,
                   {"sha": sha, "task": "E9", "op": op,
                    "instruction_template": {
                        'twist': "twist the object around its vertical axis",
                        'taper': "taper the object so it narrows towards the top",
                        'bulge': "make the middle of the object bulge outwards"}[op]})
        made['E9'] += 1

    if not parts:
        continue
    pid = int(parts[rng.randint(len(parts))][0])

    # --- E3 removal
    if made['E3'] < N_E3:
        after = apply_face_mask(meshes, fids != pid)
        if after:
            od = write_pair('E3', f"{sha}__p{pid}", meshes, after,
                            {"sha": sha, "task": "E3", "part_id": pid,
                             "instruction_template": "remove the {PART}"})
            save_part_ref(meshes, fids, pid, sha, f"{od}/part_ref.jpg")
            made['E3'] += 1

    # --- E6 rigid transform on part
    if made['E6'] < N_E6:
        tc = np.concatenate([m.triangles_center for m in meshes])
        c = tc[fids == pid].mean(axis=0)
        op = ['scale', 'rotate', 'translate'][made['E6'] % 3]
        T = np.eye(4)
        if op == 'scale':
            s = 1.45
            T[:3, :3] *= s
            T[:3, 3] = c - s * c
            desc = "make the {PART} about 1.5x larger"
        elif op == 'rotate':
            ang = np.deg2rad(30)
            R = trimesh.transformations.rotation_matrix(ang, [0, 0, 1], c)
            T = R
            desc = "rotate the {PART} by 30 degrees"
        else:
            off = np.array([0.25 * ext[0], 0, 0])
            T[:3, 3] = off
            desc = "shift the {PART} sideways"
        after = transform_part_vertices(meshes, fids, pid, T)
        od = write_pair('E6', f"{sha}__p{pid}_{op}", meshes, after,
                        {"sha": sha, "task": "E6", "part_id": pid, "op": op,
                         "instruction_template": desc})
        save_part_ref(meshes, fids, pid, sha, f"{od}/part_ref.jpg")
        made['E6'] += 1

    # --- E7 duplicate / mirror
    if made['E7'] < N_E7:
        part = extract_part(meshes, fids, pid)
        if part:
            tc = np.concatenate([m.triangles_center for m in meshes])
            c = tc[fids == pid].mean(axis=0)
            allc = tc.mean(axis=0)
            op = 'mirror' if abs(c[0] - allc[0]) > 0.12 * ext[0] else 'duplicate'
            dup = [p.copy() for p in part]
            if op == 'mirror':
                M = np.eye(4); M[0, 0] = -1
                M[0, 3] = 2 * allc[0]
                for p in dup:
                    p.apply_transform(M)
                desc = "add a mirrored copy of the {PART} on the other side"
            else:
                pext = np.ptp(np.concatenate([p.vertices for p in dup]), axis=0)
                off = np.zeros(3); ax = int(np.argmax(ext[:2]))
                off[ax] = pext[ax] * 1.15
                for p in dup:
                    p.apply_translation(off)
                desc = "add a second copy of the {PART} next to it"
            od = write_pair('E7', f"{sha}__p{pid}_{op}", meshes, meshes + dup,
                            {"sha": sha, "task": "E7", "part_id": pid, "op": op,
                             "instruction_template": desc})
            save_part_ref(meshes, fids, pid, sha, f"{od}/part_ref.jpg")
            made['E7'] += 1
    print(sha[:8], made, flush=True)

# --- E8 cross-asset composition
donors = []
for sha in cands:
    if len(donors) >= N_E8 * 2:
        break
    meshes, fids, _ = get(sha)
    if meshes is None:
        continue
    p = part_stats(meshes, fids)
    if p:
        donors.append(sha)
for i in range(0, min(len(donors) - 1, N_E8 * 2 - 1), 2):
    a_sha, b_sha = donors[i], donors[i + 1]
    A, _, _ = get(a_sha)
    Bm, Bf, _ = get(b_sha)
    pid = int(part_stats(Bm, Bf)[0][0])
    part = extract_part(Bm, Bf, pid)
    if not part:
        continue
    Av = np.concatenate([m.vertices for m in A])
    pv = np.concatenate([p.vertices for p in part])
    s = 0.35 * np.linalg.norm(np.ptp(Av, axis=0)) / max(np.linalg.norm(np.ptp(pv, axis=0)), 1e-6)
    top = np.array([Av[:, 0].mean(), Av[:, 1].mean(), Av[:, 2].max()])
    pc = pv.mean(axis=0)
    for p in part:
        p.apply_translation(-pc)
        p.apply_scale(s)
    pmin = np.concatenate([p.vertices for p in part])[:, 2].min()
    for p in part:
        p.apply_translation(top - [0, 0, pmin])
    od = write_pair('E8', f"{a_sha[:12]}__{b_sha[:12]}_p{pid}", A, A + part,
                    {"sha": a_sha, "donor": b_sha, "task": "E8", "part_id": pid,
                     "instruction_template": "attach the {PART} from the other object on top"})
    save_part_ref(Bm, Bf, pid, b_sha, f"{od}/part_ref.jpg")
    made['E8'] += 1
    print('E8', a_sha[:8], '<-', b_sha[:8], flush=True)

print("MADE:", made)
print("PROCEDURAL_DONE")
