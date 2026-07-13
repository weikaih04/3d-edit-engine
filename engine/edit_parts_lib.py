"""Shared lib for procedural round-2: area-based part stats + z-buffer visibility.
Verified invariant: P3-SAM face_ids map 1:1 onto trimesh scene-dump merged face order.
"""
import os, json, glob
import numpy as np
import trimesh

B = "/fsx/hyperpod/weikaih_edit"
_pilot = {a['sha']: a for a in json.load(open(f'{B}/pilot200.json'))}


def pilot():
    return _pilot


def load_dump(sha):
    sc = trimesh.load(_pilot[sha]['glb'])
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


def sample_with_parts(meshes, fids, n=30000):
    merged = trimesh.util.concatenate([m.copy() for m in meshes])
    pts, fi = trimesh.sample.sample_surface(merged, n)
    return pts, fids[fi]


def visibility(pts, pid_of_pts, pid, res=128):
    """max over 6 axis-aligned ortho views of: fraction of the part's projected
    cells where the part point is FRONTMOST (z-buffer). ~ 'can you see this part'."""
    hot = pid_of_pts == pid
    if not hot.any():
        return 0.0
    lo, hi = pts.min(0), pts.max(0)
    span = np.maximum(hi - lo, 1e-9)
    q = ((pts - lo) / span * (res - 1)).astype(int)
    best = 0.0
    for ax in range(3):
        u, v = [i for i in range(3) if i != ax]
        for sign in (1, -1):
            depth = pts[:, ax] * sign
            cell = q[:, u] * res + q[:, v]
            order = np.argsort(-depth)          # frontmost first
            c_sorted = cell[order]
            first = np.unique(c_sorted, return_index=True)[1]
            front_idx = order[first]            # frontmost point per occupied cell
            front_hot = hot[front_idx]
            part_cells = np.unique(cell[hot])
            if len(part_cells) == 0:
                continue
            vis = front_hot.sum() / len(part_cells)
            best = max(best, float(min(vis, 1.0)))
    return best


def part_stats_v2(meshes, fids, min_share=0.03, max_share=0.35,
                  min_vis=0.30, min_faces=60, pts_cache=None):
    """usable parts by SURFACE AREA share + visibility; excludes the base part.
    returns list of dicts sorted by area share desc."""
    areas = np.concatenate([m.area_faces for m in meshes])
    tot = areas.sum()
    uniq = np.unique(fids)
    tc = np.concatenate([m.triangles_center for m in meshes])
    zmin = {u: tc[fids == u][:, 2].min() for u in uniq}
    base = min(zmin, key=zmin.get)
    if pts_cache is None:
        pts_cache = sample_with_parts(meshes, fids)
    pts, ppid = pts_cache
    out = []
    for u in uniq:
        if u == base:
            continue
        mask = fids == u
        if mask.sum() < min_faces:
            continue
        share = areas[mask].sum() / tot
        if not (min_share <= share <= max_share):
            continue
        vis = visibility(pts, ppid, u)
        if vis < min_vis:
            continue
        out.append({"pid": int(u), "area_share": float(share), "vis": float(vis),
                    "n_faces": int(mask.sum())})
    return sorted(out, key=lambda x: -x['area_share']), (pts, ppid)


def export_scene(meshes, path):
    sc = trimesh.Scene()
    for i, m in enumerate(meshes):
        sc.add_geometry(m, node_name=f"n{i}", geom_name=f"g{i}")
    sc.export(path)


def apply_face_mask(meshes, keep):
    out = []
    for m, (a, b) in zip(meshes, face_slices(meshes)):
        mm = m.copy()
        k = keep[a:b]
        if k.all():
            out.append(mm); continue
        if not k.any():
            continue
        mm.update_faces(k); mm.remove_unreferenced_vertices()
        if len(mm.faces):
            out.append(mm)
    return out


def transform_part_vertices(meshes, fids, pid, T):
    out = []
    for m, (a, b) in zip(meshes, face_slices(meshes)):
        mm = m.copy()
        pf = np.where(fids[a:b] == pid)[0]
        if len(pf):
            vids = np.unique(mm.faces[pf])
            v = mm.vertices[vids]
            mm.vertices[vids] = (np.c_[v, np.ones(len(v))] @ T.T)[:, :3]
        out.append(mm)
    return out


def save_part_ref(meshes, fids, pid, sha, path, pts_cache=None):
    """part ref for VLM naming: VISIBLE-ONLY points (z-buffer front layer),
    part in red — occluded interior no longer pollutes the view."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    if pts_cache is None:
        pts_cache = sample_with_parts(meshes, fids)
    pts, ppid = pts_cache
    hot = ppid == pid
    res = 200
    lo, hi = pts.min(0), pts.max(0)
    span = np.maximum(hi - lo, 1e-9)
    q = ((pts - lo) / span * (res - 1)).astype(int)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    axes[0].imshow(Image.open(_pilot[sha]['front_view']).convert('RGB'))
    axes[0].set_title('asset'); axes[0].axis('off')
    views = [(1, 0, 2, 'front (-y)'), (0, 1, 2, 'side (-x)'), (2, 0, 1, 'top (+z)')]
    for ax_i, (ax, u, v, nm) in enumerate(views):
        depth = pts[:, ax] * (1 if nm.startswith('top') else -1)
        cell = q[:, u] * res + q[:, v]
        order = np.argsort(-depth)
        first = np.unique(cell[order], return_index=True)[1]
        fi = order[first]
        p, h = pts[fi], hot[fi]
        a = axes[ax_i + 1]
        a.scatter(p[~h][:, u], p[~h][:, v], c='lightgray', s=0.7)
        a.scatter(p[h][:, u], p[h][:, v], c='red', s=1.0)
        a.set_aspect('equal'); a.axis('off'); a.set_title(nm)
    plt.tight_layout(); plt.savefig(path, dpi=100); plt.close()


def part_gate(sha):
    """asset-level gate for part ops: judge part_complexity >= 4"""
    return _pilot[sha].get('part_complexity', 0) >= 4
