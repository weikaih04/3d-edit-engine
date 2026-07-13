"""Render after.glb outputs with Blender CPU (CYCLES), 4 fixed views,
then build side-by-side before/after comparison grids.
Usage: python render_outputs.py <out_pilot_dir> [n_parallel]
Grid: [orig render 000] [after view0] [after view1] | per asset row saved to compare/
"""
import os, sys, json, glob, subprocess
from concurrent.futures import ThreadPoolExecutor

BLENDER = "/tmp/blender-4.5.1-linux-x64/blender"
SCRIPT = "/fsx/sfr/weikaih/trellis2-data-pipeline/toolkit/blender_script/render_cond.py"
BASE = sys.argv[1] if len(sys.argv) > 1 else "/fsx/hyperpod/weikaih_edit/out_pilot/null"
NPAR = int(sys.argv[2]) if len(sys.argv) > 2 else 8

# fov is in RADIANS (script convention: fov = 2*arcsin(sqrt(3)/2/radius))
import math
_R = 2.0
_FOV = 2 * math.asin(math.sqrt(3) / 2 / _R)
views = [
    {"yaw": 0.0, "pitch": 0.35, "radius": _R, "fov": _FOV},
    {"yaw": 2.094, "pitch": 0.35, "radius": _R, "fov": _FOV},
    {"yaw": 4.189, "pitch": 0.35, "radius": _R, "fov": _FOV},
]

def render_one(d):
    glb = f"{d}/after.glb"
    if not os.path.exists(glb) or os.path.exists(f"{d}/render_000.webp"):
        return d, os.path.exists(f"{d}/render_000.webp")
    cmd = [BLENDER, "-b", "-t", "8", "-P", SCRIPT, "--",
           "--object", glb,
           "--cond_views", json.dumps(views),
           "--cond_output_folder", d,
           "--cond_resolution", "512",
           "--engine", "CYCLES"]
    env = {**os.environ, "OMP_NUM_THREADS": "8",
           "LD_LIBRARY_PATH": "/fsx/sfr/weikaih/trellis2-data-pipeline/env/blender_libs:"
                              + os.environ.get("LD_LIBRARY_PATH", "")}
    r = subprocess.run(cmd, capture_output=True, timeout=900, env=env)
    outs = sorted(glob.glob(f"{d}/0*.webp")) + sorted(glob.glob(f"{d}/0*.png"))
    for i, o in enumerate(outs):
        os.rename(o, f"{d}/render_{i:03d}" + os.path.splitext(o)[1].replace('png','webp'))
    return d, len(outs) > 0

dirs = sorted(glob.glob(f"{BASE}/*/"))
print(f"{len(dirs)} outputs to render")
with ThreadPoolExecutor(NPAR) as ex:
    for d, ok in ex.map(render_one, dirs):
        print(os.path.basename(d.rstrip('/')), 'OK' if ok else 'FAIL', flush=True)

# comparison grids
from PIL import Image
cmp_dir = f"{BASE}/../compare_{os.path.basename(BASE.rstrip('/'))}"
os.makedirs(cmp_dir, exist_ok=True)
pilot = {a['sha']: a for a in json.load(open('/fsx/hyperpod/weikaih_edit/pilot200.json'))}
n = 0
for d in dirs:
    sha = os.path.basename(d.rstrip('/'))
    rend = sorted(glob.glob(f"{d}/render_*"))
    if sha not in pilot or not rend:
        continue
    def load_sq(p):
        im = Image.open(p)
        if im.mode == 'RGBA':
            bg = Image.new('RGBA', im.size, (40, 40, 40, 255))
            im = Image.alpha_composite(bg, im)
        return im.convert('RGB').resize((512, 512), Image.LANCZOS)
    orig = load_sq(pilot[sha]['front_view'])
    cond = f"{d}/cond_image.png"
    row = [orig]
    if os.path.exists(cond):
        row.append(load_sq(cond))
    row += [load_sq(r) for r in rend[:3]]
    W = sum(im.width for im in row)
    grid = Image.new('RGB', (W, 512), 'white')
    x = 0
    for im in row:
        grid.paste(im, (x, 0)); x += im.width
    grid.save(f"{cmp_dir}/{sha}.jpg", quality=90)
    n += 1
print(f"grids: {n} -> {cmp_dir}")
