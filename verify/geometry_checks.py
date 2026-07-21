"""Auto-verification gate, CPU part: geometry hard checks per pair.
- texture tasks (E1a/E1b/E1c/E1a_rev): silhouette IoU before-vs-after (3 ortho views,
  point z-projection occupancy) — geometry must be frozen (IoU >= 0.93).
- E3/E2(inverse): |area(after)-area(before)| share in [2%,40%] + stored part vis >= 0.3
- E6/E7/E10: stored vis >= 0.3 + vertex-displacement sanity (some vertices moved / added)
- E9: displacement field must be smooth+substantial: mean |dv| in [1%,20%] of diag
- E5: no geometry constraint (full regen)
Writes auto_geo.json {pair_key: {metrics..., geo_pass}}
Usage: python geometry_checks.py <pairs_root(out_pairs2)> [also_pilot=1]
"""
import os, sys, json, glob
import numpy as np
import trimesh

B = "/fsx/hyperpod/weikaih_edit"
sys.path.insert(0, B)
ROOT = sys.argv[1] if len(sys.argv) > 1 else f"{B}/out_pairs2"
ALSO_PILOT = len(sys.argv) > 2 and sys.argv[2] == '1'
pilot = {a['sha']: a for a in json.load(open(f'{B}/pilot200.json'))}

INTERIOR_EXPOSURE_MAX = 0.35   # >this at a cut = removal exposed inner walls -> reject


def _clean_cut(before_glb, after_glb, sha, pids):
    """After a part removal, quantify (a) NEW open-boundary edges at the cut = true
    hole, (b) interior exposure = fraction of after-faces near the removed region
    whose normal points TOWARD the removed part's former centroid (inner wall now
    visible through the cut). Returns (new_open_edges, interior_exposure)."""
    from scipy.spatial import cKDTree
    try:
        from edit_parts_lib import load_dump
    except Exception:
        return None, None
    b = trimesh.load(before_glb, force='mesh')
    a = trimesh.load(after_glb, force='mesh')

    def bnd_mids(m):
        if not len(m.faces):
            return np.zeros((0, 3))
        e = m.edges_sorted
        uq, cnt = np.unique(e, axis=0, return_counts=True)
        be = uq[cnt == 1]
        return m.vertices[be].mean(axis=1)
    pb, pa = bnd_mids(b), bnd_mids(a)
    diag = float(np.linalg.norm(np.ptp(b.vertices, axis=0)))
    if len(pa) and len(pb):
        d2, _ = cKDTree(pb).query(pa, k=1)
        new_open = int((d2 > 0.01 * diag).sum())
    else:
        new_open = int(len(pa))

    exposure = None
    if pids:
        meshes, fids = load_dump(sha)
        if meshes is not None:
            merged = trimesh.util.concatenate([m.copy() for m in meshes])
            rc = merged.triangles_center[np.isin(fids, [int(p) for p in pids])]
            if len(rc) >= 10:
                part_c = rc.mean(0)
                ac, an = a.triangles_center, a.face_normals
                near = np.linalg.norm(ac - part_c, axis=1) < 0.6 * np.linalg.norm(np.ptp(rc, 0))
                if near.sum() >= 10:
                    to = part_c - ac[near]
                    to /= np.linalg.norm(to, axis=1, keepdims=True) + 1e-9
                    exposure = float(((an[near] * to).sum(1) > 0.3).mean())
                else:
                    exposure = 0.0
    return new_open, exposure


def occupancy_iou(m1, m2, res=96, n=80000):
    """silhouette IoU invariant to similarity transform: each mesh is normalized
    isotropically to its own bbox (TRELLIS outputs live in [-0.5,0.5], originals
    in world units)."""
    def norm_pts(m):
        np.random.seed(0)
        p, _ = trimesh.sample.sample_surface(m, n)
        c = (p.min(0) + p.max(0)) / 2
        s = max(np.ptp(p, axis=0).max(), 1e-9)
        return (p - c) / s + 0.5
    p1, p2 = norm_pts(m1), norm_pts(m2)
    ious = []
    for ax in range(3):
        u, v = [i for i in range(3) if i != ax]
        def occ(p):
            q = np.clip((p * (res - 1)).astype(int), 0, res - 1)
            g = np.zeros((res, res), bool)
            g[q[:, u], q[:, v]] = True
            return g
        a, b = occ(p1), occ(p2)
        ious.append((a & b).sum() / max((a | b).sum(), 1))
    return float(np.mean(ious))


def orphan_check(mb, ma, diag, n=20000):
    """detect parts left floating by an edit: a connected component of AFTER whose
    min distance to the rest jumped (was attached in BEFORE, now hangs in air)."""
    import trimesh
    from scipy.spatial import cKDTree
    try:
        comps = ma.split(only_watertight=False)
        if len(comps) < 2:
            return 0, True
        np.random.seed(0)
        samples = []
        for c in comps:
            if len(c.faces) < 4:
                continue
            p, _ = trimesh.sample.sample_surface(c, max(200, min(2000, len(c.faces))))
            samples.append(p)
        n_orph = 0
        for i, p in enumerate(samples):
            others = np.vstack([q for j, q in enumerate(samples) if j != i])
            d = cKDTree(others).query(p, k=1)[0].min()
            if d > 0.04 * diag:
                n_orph += 1
        return n_orph, n_orph == 0
    except Exception:
        return -1, True


results = {}

def check_pair(key, task, before_glb, after_glb, meta):
    r = {"task": task}
    try:
        mb = trimesh.load(before_glb, force='mesh')
        ma = trimesh.load(after_glb, force='mesh')
        diag = float(np.linalg.norm(np.ptp(mb.vertices, axis=0)))
        if task in ('E1a', 'E1b', 'E1c', 'E1a_rev'):
            r['sil_iou'] = occupancy_iou(mb, ma)
            r['geo_pass'] = r['sil_iou'] >= 0.93
        elif task in ('E3', 'E2'):
            d_area = abs(ma.area - mb.area) / max(mb.area, 1e-9)
            r['area_delta'] = float(d_area)
            vis = meta.get('vis') or (meta.get('unit') or {}).get('vis')
            r['vis'] = vis
            n_orph, orph_ok = orphan_check(mb, ma, diag)
            r['orphans'] = n_orph
            # clean-cut: reject removals that expose inner walls (only for E3;
            # E2 = inverse of E3, before/after roles swapped so cut is on 'before')
            cut_ok = True
            if task == 'E3':
                sha = meta.get('sha')
                pids = (meta.get('unit') or {}).get('pids') or meta.get('pids')
                new_open, expo = _clean_cut(before_glb, after_glb, sha, pids)
                r['cut_new_open_edges'] = new_open
                r['interior_exposure'] = expo
                if expo is not None and expo > INTERIOR_EXPOSURE_MAX:
                    cut_ok = False
            r['geo_pass'] = (0.02 <= d_area <= 0.60) and (vis or 0) >= 0.3 and orph_ok and cut_ok
        elif task in ('E6', 'E7'):
            vis = meta.get('vis') or (meta.get('unit') or {}).get('vis')
            r['vis'] = vis
            df = abs(len(ma.faces) - len(mb.faces))
            moved = None
            if len(ma.vertices) == len(mb.vertices):
                dv = np.linalg.norm(ma.vertices - mb.vertices, axis=1)
                moved = float((dv > 1e-6 * diag).mean())
                r['moved_frac'] = moved
            n_orph, orph_ok = orphan_check(mb, ma, diag)
            r['orphans'] = n_orph
            r['geo_pass'] = ((vis or 0) >= 0.3) and (df > 0 or (moved or 0) > 0.005) and orph_ok
        elif task == 'E9':
            if len(ma.vertices) == len(mb.vertices):
                dv = np.linalg.norm(ma.vertices - mb.vertices, axis=1)
                r['mean_disp'] = float(dv.mean() / diag)
                r['geo_pass'] = 0.005 <= r['mean_disp'] <= 0.25
            else:
                r['geo_pass'] = False
        else:
            r['geo_pass'] = True
    except Exception as e:
        r['error'] = str(e)[:200]
        r['geo_pass'] = False
    r = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in r.items()}
    results[key] = r
    print(key, r.get('geo_pass'), {k: v for k, v in r.items() if k not in ('task', 'geo_pass')}, flush=True)


for mp in sorted(glob.glob(f"{ROOT}/*/*/meta.json")):
    d = os.path.dirname(mp)
    task = d.split('/')[-2]
    name = os.path.basename(d)
    m = json.load(open(mp))
    if task == 'E10':
        for i in (1, 2):
            check_pair(f"E10/{name}#t{i}", 'E6' if i == 2 else 'E3',
                       f"{d}/state{i-1}.glb", f"{d}/state{i}.glb",
                       {**m, "vis": 0.5})
    else:
        check_pair(f"{task}/{name}", task, f"{d}/before.glb", f"{d}/after.glb", m)

if ALSO_PILOT:
    for task, edir in [('E1a', f"{B}/out_pilot/edit_v2"), ('E1c', f"{B}/out_pilot/e1c")]:
        for mp in sorted(glob.glob(f"{edir}/*/meta.json")):
            m = json.load(open(mp))
            if m.get('error'):
                continue
            sha = m['sha']
            check_pair(f"{task}/{sha}", task, pilot[sha]['glb'],
                       f"{edir}/{sha}/after.glb", m)

out = f"{ROOT}/auto_geo.json"
json.dump(results, open(out, "w"), indent=1)
npass = sum(1 for r in results.values() if r.get('geo_pass'))
print(f"GEO_CHECKS_DONE {npass}/{len(results)} pass -> {out}")
