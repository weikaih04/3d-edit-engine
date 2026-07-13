"""Part-closeup ('look-at-part') renders for E2/E7 pairs: anchor the render
framing to the edited part's bbox (x2.2 margin) via invisible micro-triangles,
so the camera frames the edit region instead of the whole object.
Writes before_z000/after_z000 webps next to the pair glbs.
Usage: python render_zoom.py [n_parallel]
"""
import os, sys, json, glob, subprocess, math
import numpy as np
import trimesh
from concurrent.futures import ThreadPoolExecutor

B = "/fsx/hyperpod/weikaih_edit"
BLENDER = "/tmp/blender-4.5.1-linux-x64/blender"
SCRIPT = "/fsx/sfr/weikaih/trellis2-data-pipeline/toolkit/blender_script/render_cond.py"
NPAR = int(sys.argv[1]) if len(sys.argv) > 1 else 8
_R = 2.0
_FOV = 2 * math.asin(math.sqrt(3) / 2 / _R)
views = [{"yaw": 0.9, "pitch": 0.35, "radius": _R, "fov": _FOV}]
pilot = {a['sha']: a for a in json.load(open(f'{B}/pilot200.json'))}

jobs = []
for task in ['E2', 'E7']:
    for mp in sorted(glob.glob(f"{B}/out_pairs2/{task}/*/meta.json")):
        d = os.path.dirname(mp) + '/'
        m = json.load(open(mp))
        sha, pid = m['sha'], m.get('part_id')
        if pid is None:
            continue
        fp = f"{B}/out_p3sam/all/{sha}/face_ids.npy"
        if not os.path.exists(fp):
            continue
        for g in ['before', 'after']:
            if not os.path.exists(f"{d}{g}_z000.webp"):
                jobs.append((d, g, sha, pid))
print(f"{len(jobs)} zoom renders", flush=True)


def part_box(sha, pid):
    orig = trimesh.load(pilot[sha]['glb'], force='mesh')
    fids = np.load(f"{B}/out_p3sam/all/{sha}/face_ids.npy")
    sel = orig.triangles_center[fids == pid]
    lo, hi = sel.min(0), sel.max(0)
    c, ext = (lo + hi) / 2, (hi - lo)
    half = max(ext.max() * 1.1, np.ptp(orig.vertices, axis=0).max() * 0.15)
    return c - half, c + half


def rone(j):
    d, g, sha, pid = j
    try:
        lo, hi = part_box(sha, pid)
        eps = float(np.linalg.norm(hi - lo)) * 5e-4
        sc = trimesh.load(f"{d}{g}.glb")
        if not isinstance(sc, trimesh.Scene):
            sc = trimesh.Scene(sc)
        for i, c in enumerate([lo, hi]):
            t = trimesh.Trimesh(vertices=[c, c + [eps, 0, 0], c + [0, eps, 0]], faces=[[0, 1, 2]])
            sc.add_geometry(t, node_name=f"_z{i}", geom_name=f"_z{i}")
        zglb = f"{d}{g}_zoom.glb"
        sc.export(zglb)
        tmp = f"{d}_rz_{g}"
        os.makedirs(tmp, exist_ok=True)
        env = {**os.environ, "OMP_NUM_THREADS": "8",
               "LD_LIBRARY_PATH": "/fsx/sfr/weikaih/trellis2-data-pipeline/env/blender_libs:" + os.environ.get("LD_LIBRARY_PATH", "")}
        subprocess.run([BLENDER, "-b", "-t", "8", "-P", SCRIPT, "--", "--object", zglb,
                        "--cond_views", json.dumps(views), "--cond_output_folder", tmp,
                        "--cond_resolution", "512", "--engine", "CYCLES"],
                       capture_output=True, timeout=900, env=env)
        outs = sorted(glob.glob(f"{tmp}/0*.webp"))
        for i, o in enumerate(outs):
            os.replace(o, f"{d}{g}_z{i:03d}.webp")
        try:
            os.rmdir(tmp)
        except OSError:
            pass
        os.remove(zglb)
        return d + g, len(outs) > 0
    except Exception as e:
        return d + g, False


with ThreadPoolExecutor(NPAR) as ex:
    for k, ok in ex.map(rone, jobs):
        print(k.split('/')[-2], k.split('/')[-1] if '/' in k else '', 'OK' if ok else 'FAIL', flush=True)
print("RENDER_ZOOM_DONE")
