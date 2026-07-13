"""Rebuild comparison grids only (LANCZOS + neutral bg), no re-render."""
import os, sys, glob, json
from PIL import Image

BASE = sys.argv[1] if len(sys.argv) > 1 else "/fsx/hyperpod/weikaih_edit/out_pilot/null"
cmp_dir = f"{os.path.dirname(BASE.rstrip('/'))}/compare_{os.path.basename(BASE.rstrip('/'))}"
os.makedirs(cmp_dir, exist_ok=True)
pilot = {a['sha']: a for a in json.load(open('/fsx/hyperpod/weikaih_edit/pilot200.json'))}


def load_sq(p):
    im = Image.open(p)
    if im.mode == 'RGBA':
        bg = Image.new('RGBA', im.size, (40, 40, 40, 255))
        im = Image.alpha_composite(bg, im)
    return im.convert('RGB').resize((512, 512), Image.LANCZOS)


n = 0
for d in sorted(glob.glob(f"{BASE}/*/")):
    sha = os.path.basename(d.rstrip('/'))
    rend = sorted(glob.glob(f"{d}/render_*"))
    if sha not in pilot or not rend:
        continue
    row = [load_sq(pilot[sha]['front_view'])]
    if os.path.exists(f"{d}/cond_image.png"):
        row.append(load_sq(f"{d}/cond_image.png"))
    row += [load_sq(r) for r in rend[:3]]
    W = sum(im.width for im in row)
    grid = Image.new('RGB', (W, 512), 'white')
    x = 0
    for im in row:
        grid.paste(im, (x, 0)); x += im.width
    grid.save(f"{cmp_dir}/{sha}.jpg", quality=92)
    n += 1
print(f"rebuilt {n} grids -> {cmp_dir}")
