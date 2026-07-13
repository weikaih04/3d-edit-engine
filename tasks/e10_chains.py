"""E10 multi-turn chains: compose 2 procedural edits sequentially on one asset.
turn1 = remove part p1;  turn2 = transform (scale/rotate) part p2  (p1 != p2).
Saves state0/state1/state2 glbs + per-turn instruction templates + part refs.
Usage: python e10_chains.py [n_chains]
"""
import os, sys, json, glob
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

B = "/fsx/hyperpod/weikaih_edit"
OUT = f"{B}/out_pairs/E10"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
rng = np.random.RandomState(23)
pilot = {a['sha']: a for a in json.load(open(f'{B}/pilot200.json'))}


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
        out.append((off, off + len(m.faces))); off += len(m.faces)
    return out


def part_stats(meshes, fids):
    uniq, cnt = np.unique(fids, return_counts=True)
    tc = np.concatenate([m.triangles_center for m in meshes])
    zmin = {u: tc[fids == u][:, 2].min() for u in uniq}
    base = min(zmin, key=zmin.get)
    share = cnt / cnt.sum()
    return sorted([(u, c) for u, c, s in zip(uniq, cnt, share)
                   if 0.03 <= s <= 0.35 and u != base and c > 60], key=lambda x: -x[1])


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


cands = [os.path.basename(d.rstrip('/')) for d in sorted(glob.glob(f"{B}/out_p3sam/all/*/"))]
cands = [s for s in cands if s in pilot]
rng.shuffle(cands)

made = 0
for sha in cands:
    if made >= N:
        break
    meshes, fids = load_dump(sha)
    if meshes is None:
        continue
    parts = part_stats(meshes, fids)
    if len(parts) < 2:
        continue
    ids = rng.choice(len(parts), 2, replace=False)
    p1, p2 = int(parts[ids[0]][0]), int(parts[ids[1]][0])

    # turn 1: remove p1
    keep = fids != p1
    s1, kept_fids = [], []
    for m, (a, b) in zip(meshes, face_slices(meshes)):
        mm = m.copy(); k = keep[a:b]
        if not k.any():
            continue
        if not k.all():
            mm.update_faces(k); mm.remove_unreferenced_vertices()
        if len(mm.faces):
            s1.append(mm); kept_fids.append(fids[a:b][k])
    f1 = np.concatenate(kept_fids)

    # turn 2: scale p2 (on the state-1 meshes)
    tc = np.concatenate([m.triangles_center for m in s1])
    if (f1 == p2).sum() < 40:
        continue
    c = tc[f1 == p2].mean(axis=0)
    s = 1.45
    T = np.eye(4); T[:3, :3] *= s; T[:3, 3] = c - s * c
    s2 = []
    for m, (a, b) in zip(s1, face_slices(s1)):
        mm = m.copy()
        pf = np.where(f1[a:b] == p2)[0]
        if len(pf):
            vids = np.unique(mm.faces[pf])
            v = mm.vertices[vids]
            mm.vertices[vids] = (np.c_[v, np.ones(len(v))] @ T.T)[:, :3]
        s2.append(mm)

    od = f"{OUT}/{sha}"
    os.makedirs(od, exist_ok=True)
    export_scene(meshes, f"{od}/state0.glb")
    export_scene(s1, f"{od}/state1.glb")
    export_scene(s2, f"{od}/state2.glb")
    save_part_ref(meshes, fids, p1, sha, f"{od}/part1_ref.jpg")
    save_part_ref(meshes, fids, p2, sha, f"{od}/part2_ref.jpg")
    json.dump({"sha": sha, "task": "E10",
               "turns": [
                   {"part_id": p1, "instruction_template": "remove the {PART1}"},
                   {"part_id": p2, "instruction_template": "make the {PART2} about 1.5x larger"}]},
              open(f"{od}/meta.json", "w"), indent=1)
    made += 1
    print(sha[:8], 'chain p%d->p%d' % (p1, p2), flush=True)
print(f"E10_DONE {made}")
