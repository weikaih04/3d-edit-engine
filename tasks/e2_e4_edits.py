"""E2 part addition + E4 part replacement — procedural-exact v0.
E2: paste a donor part (from another asset) at a plausible attach point (top/side).
E4: remove a P3-SAM part, fit a donor part into the removed part's bbox.
Reuses the verified face_ids<->original-glb 1:1 mapping. 8 pairs each.
Usage: python e2_e4_edits.py
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
rng = np.random.RandomState(7)
pilot = {a['sha']: a for a in json.load(open(f'{B}/pilot200.json'))}
N_E2, N_E4 = 8, 8


def load_dump(sha):
    sc = trimesh.load(pilot[sha]['glb'])
    meshes = sc.dump() if isinstance(sc, trimesh.Scene) else [sc]
    meshes = [m for m in meshes if isinstance(m, trimesh.Trimesh) and len(m.faces)]
    d = f"{B}/out_p3sam/all/{sha}/"
    if not os.path.exists(d + 'face_ids.npy'):
        return None, None
    fids = np.load(d + 'face_ids.npy')
    if sum(len(m.faces) for m in meshes) != len(fids):
        return None, None
    return meshes, fids


def face_slices(meshes):
    out, off = [], 0
    for m in meshes:
        out.append((off, off + len(m.faces)))
        off += len(m.faces)
    return out


def apply_face_mask(meshes, keep):
    out = []
    for m, (a, b) in zip(meshes, face_slices(meshes)):
        mm = m.copy(); k = keep[a:b]
        if k.all():
            out.append(mm); continue
        if not k.any():
            continue
        mm.update_faces(k); mm.remove_unreferenced_vertices()
        if len(mm.faces):
            out.append(mm)
    return out


def part_stats(meshes, fids):
    uniq, cnt = np.unique(fids, return_counts=True)
    tc = np.concatenate([m.triangles_center for m in meshes])
    zmin = {u: tc[fids == u][:, 2].min() for u in uniq}
    base = min(zmin, key=zmin.get)
    share = cnt / cnt.sum()
    return sorted([(u, c) for u, c, s in zip(uniq, cnt, share)
                   if 0.03 <= s <= 0.35 and u != base and c > 60],
                  key=lambda x: -x[1])


def export_scene(meshes, path):
    sc = trimesh.Scene()
    for i, m in enumerate(meshes):
        sc.add_geometry(m, node_name=f"n{i}", geom_name=f"g{i}")
    sc.export(path)


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


def write_pair(task, name, before, after, meta):
    od = f"{OUT}/{task}/{name}"
    os.makedirs(od, exist_ok=True)
    export_scene(before, f"{od}/before.glb")
    export_scene(after, f"{od}/after.glb")
    json.dump(meta, open(f"{od}/meta.json", "w"), indent=1)
    return od


cands = []
for d in sorted(glob.glob(f"{B}/out_p3sam/all/*/")):
    sha = os.path.basename(d.rstrip('/'))
    if sha in pilot:
        cands.append(sha)
rng.shuffle(cands)

# collect (sha, pid) donors with usable parts
loaded, donors = {}, []
def get(sha):
    if sha not in loaded:
        loaded[sha] = load_dump(sha)
    return loaded[sha]

for sha in cands:
    m, f = get(sha)
    if m is None:
        continue
    p = part_stats(m, f)
    if p:
        donors.append((sha, int(p[0][0])))
    if len(donors) >= (N_E2 + N_E4) * 2:
        break

made = {'E2': 0, 'E4': 0}
ptr = 0
for sha in cands:
    if made['E2'] >= N_E2 and made['E4'] >= N_E4:
        break
    meshes, fids = get(sha)
    if meshes is None:
        continue
    # pick a donor from a DIFFERENT asset
    while ptr < len(donors) and donors[ptr][0] == sha:
        ptr += 1
    if ptr >= len(donors):
        break
    d_sha, d_pid = donors[ptr]; ptr += 1
    dm, df = get(d_sha)
    part = apply_face_mask(dm, df == d_pid)
    if not part:
        continue
    Av = np.concatenate([m.vertices for m in meshes])
    pv = np.concatenate([p.vertices for p in part])
    pc = pv.mean(axis=0)

    if made['E2'] <= made['E4'] and made['E2'] < N_E2:
        # E2: attach donor part on top, scaled to 30% of asset diag
        s = 0.30 * np.linalg.norm(np.ptp(Av, axis=0)) / max(np.linalg.norm(np.ptp(pv, axis=0)), 1e-6)
        dup = [p.copy() for p in part]
        for p in dup:
            p.apply_translation(-pc); p.apply_scale(s)
        zmin = np.concatenate([p.vertices for p in dup])[:, 2].min()
        top = np.array([Av[:, 0].mean(), Av[:, 1].mean(), Av[:, 2].max()])
        for p in dup:
            p.apply_translation(top - [0, 0, zmin] + [0, 0, -0.02 * np.ptp(Av[:, 2])])
        od = write_pair('E2', f"{sha[:12]}__add_{d_sha[:8]}p{d_pid}", meshes, meshes + dup,
                        {"sha": sha, "donor": d_sha, "donor_part": d_pid, "task": "E2",
                         "instruction_template": "add a {DONOR_PART} on top of the object"})
        save_part_ref(dm, df, d_pid, d_sha, f"{od}/part_ref.jpg")
        made['E2'] += 1
    elif made['E4'] < N_E4:
        # E4: replace one of THIS asset's parts with the donor part fitted to its bbox
        p_own = part_stats(meshes, fids)
        if not p_own:
            continue
        pid = int(p_own[rng.randint(len(p_own))][0])
        removed = apply_face_mask(meshes, fids != pid)
        if not removed:
            continue
        tc = np.concatenate([m.triangles_center for m in meshes])
        sel = tc[fids == pid]
        bb_c, bb_ext = sel.mean(axis=0), np.ptp(sel, axis=0)
        dup = [p.copy() for p in part]
        s = 0.9 * np.linalg.norm(bb_ext) / max(np.linalg.norm(np.ptp(pv, axis=0)), 1e-6)
        for p in dup:
            p.apply_translation(-pc); p.apply_scale(s); p.apply_translation(bb_c)
        od = write_pair('E4', f"{sha[:12]}__p{pid}_to_{d_sha[:8]}p{d_pid}", meshes, removed + dup,
                        {"sha": sha, "part_id": pid, "donor": d_sha, "donor_part": d_pid,
                         "task": "E4",
                         "instruction_template": "replace the {PART} with a {DONOR_PART}"})
        save_part_ref(meshes, fids, pid, sha, f"{od}/part_ref.jpg")
        save_part_ref(dm, df, d_pid, d_sha, f"{od}/donor_ref.jpg")
        made['E4'] += 1
    print(sha[:8], made, flush=True)

print("MADE:", made)
print("E2E4_DONE")
