"""Blender CPU renders for out_pairs: before/after (E10: state0/1/2, E1b: after only).
2 fixed views per glb. Usage: python render_pairs.py [n_parallel]
"""
import os, sys, json, glob, subprocess
from concurrent.futures import ThreadPoolExecutor
import math

B = "/fsx/hyperpod/weikaih_edit"
BLENDER = "/tmp/blender-4.5.1-linux-x64/blender"
SCRIPT = "/fsx/sfr/weikaih/trellis2-data-pipeline/toolkit/blender_script/render_cond.py"
NPAR = int(sys.argv[1]) if len(sys.argv) > 1 else 12

_R = 2.0
_FOV = 2 * math.asin(math.sqrt(3) / 2 / _R)
views = [
    {"yaw": 0.6, "pitch": 0.4, "radius": _R, "fov": _FOV},
    {"yaw": 2.7, "pitch": 0.4, "radius": _R, "fov": _FOV},
]

jobs = []
for d in sorted(glob.glob(f"{os.environ.get('PAIRS_ROOT', B + '/out_pairs')}/*/*/")):
    task = d.split('/')[-3]
    globs = ['state0', 'state1', 'state2'] if task == 'E10' else \
            (['after'] if task == 'E1b' else ['before', 'after'])
    for g in globs:
        glb = f"{d}{g}.glb"
        if os.path.exists(glb) and not os.path.exists(f"{d}{g}_r000.webp"):
            jobs.append((d, g, glb))
print(f"{len(jobs)} glbs to render", flush=True)

def anchored_glb(d, g, glb):
    """re-export glb with two invisible micro-triangles at the PAIR-UNION bbox
    corners so before/after normalize to the same framing in the render script"""
    import numpy as np
    import trimesh
    bpath = f"{d}_bounds.json"
    if not os.path.exists(bpath):
        lo, hi = None, None
        for f in glob.glob(f"{d}*.glb"):
            if f.endswith('_rn.glb'):
                continue
            try:
                m = trimesh.load(f)
                b = m.bounds if not isinstance(m, trimesh.Scene) else m.bounds
            except Exception:
                continue
            if b is None:
                continue
            lo = b[0] if lo is None else np.minimum(lo, b[0])
            hi = b[1] if hi is None else np.maximum(hi, b[1])
        json.dump([lo.tolist(), hi.tolist()], open(bpath, 'w'))
    lo, hi = json.load(open(bpath))
    import numpy as np
    lo, hi = np.array(lo), np.array(hi)
    eps = float(np.linalg.norm(hi - lo)) * 5e-4
    tris = []
    for c in [lo, hi]:
        tris.append(trimesh.Trimesh(
            vertices=[c, c + [eps, 0, 0], c + [0, eps, 0]], faces=[[0, 1, 2]]))
    sc = trimesh.load(glb)
    if not isinstance(sc, trimesh.Scene):
        sc = trimesh.Scene(sc)
    for i, t in enumerate(tris):
        sc.add_geometry(t, node_name=f"_anchor{i}", geom_name=f"_anchor{i}")
    out = f"{d}{g}_rn.glb"
    sc.export(out)
    return out


def render_one(job):
    d, g, glb = job
    try:
        glb = anchored_glb(d, g, glb)
    except Exception:
        pass  # fall back to raw glb
    tmp = f"{d}_r_{g}"
    os.makedirs(tmp, exist_ok=True)
    cmd = [BLENDER, "-b", "-t", "8", "-P", SCRIPT, "--",
           "--object", glb, "--cond_views", json.dumps(views),
           "--cond_output_folder", tmp, "--cond_resolution", "512",
           "--engine", "CYCLES"]
    env = {**os.environ, "OMP_NUM_THREADS": "8",
           "LD_LIBRARY_PATH": "/fsx/sfr/weikaih/trellis2-data-pipeline/env/blender_libs:"
                              + os.environ.get("LD_LIBRARY_PATH", "")}
    try:
        subprocess.run(cmd, capture_output=True, timeout=900, env=env)
    except subprocess.TimeoutExpired:
        return glb, False
    outs = sorted(glob.glob(f"{tmp}/0*.webp")) + sorted(glob.glob(f"{tmp}/0*.png"))
    for i, o in enumerate(outs):
        os.replace(o, f"{d}{g}_r{i:03d}.webp")
    try:
        os.rmdir(tmp)
    except OSError:
        pass
    if glb.endswith('_rn.glb') and os.path.exists(glb):
        os.remove(glb)
    return glb, len(outs) > 0

with ThreadPoolExecutor(NPAR) as ex:
    done = 0
    for glb, ok in ex.map(render_one, jobs):
        done += 1
        if done % 20 == 0 or not ok:
            print(f"{done}/{len(jobs)} {'OK' if ok else 'FAIL ' + glb}", flush=True)
print("RENDER_PAIRS_DONE", flush=True)
