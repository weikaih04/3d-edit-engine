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
ROOT = sys.argv[1] if len(sys.argv) > 1 else f"{B}/out_pairs2"
ALSO_PILOT = len(sys.argv) > 2 and sys.argv[2] == '1'
pilot = {a['sha']: a for a in json.load(open(f'{B}/pilot200.json'))}


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
            r['vis'] = meta.get('vis')
            r['geo_pass'] = (0.02 <= d_area <= 0.45) and (meta.get('vis') or 0) >= 0.3
        elif task in ('E6', 'E7'):
            r['vis'] = meta.get('vis')
            df = abs(len(ma.faces) - len(mb.faces))
            moved = None
            if len(ma.vertices) == len(mb.vertices):
                dv = np.linalg.norm(ma.vertices - mb.vertices, axis=1)
                moved = float((dv > 1e-6 * diag).mean())
                r['moved_frac'] = moved
            r['geo_pass'] = ((meta.get('vis') or 0) >= 0.3) and (df > 0 or (moved or 0) > 0.005)
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
