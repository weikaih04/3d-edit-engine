"""Final E1a QC grids: [before render][QIE-edited view][after render x3]
with the instruction as a text banner. Only assets that actually have an
after.glb. Output: out_pilot/compare_e1a/<sha>.jpg
"""
import os, glob, json
from PIL import Image, ImageDraw, ImageFont

B = "/fsx/hyperpod/weikaih_edit"
EDIT_DIR = os.environ.get("G_EDIT_DIR", f"{B}/out_pilot/edit")
VIEWS_DIR = os.environ.get("G_VIEWS_DIR", f"{B}/edited_views")
OUT = os.environ.get("G_OUT", f"{B}/out_pilot/compare_e1a")
TAG = os.environ.get("G_TAG", "E1a")
os.makedirs(OUT, exist_ok=True)
pilot = {a['sha']: a for a in json.load(open(f'{B}/pilot200.json'))}

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
except Exception:
    font = ImageFont.load_default()

def load_sq(p, size=448):
    im = Image.open(p)
    if im.mode == 'RGBA':
        bg = Image.new('RGBA', im.size, (40, 40, 40, 255))
        im = Image.alpha_composite(bg, im)
    return im.convert('RGB').resize((size, size), Image.LANCZOS)

n = 0
for d in sorted(glob.glob(f"{EDIT_DIR}/*/")):
    sha = os.path.basename(d.rstrip('/'))
    rend = sorted(glob.glob(f"{d}/render_*"))
    ij = f"{VIEWS_DIR}/{sha}.json"
    if sha not in pilot or not rend or not os.path.exists(f"{d}/after.glb"):
        continue
    instr = json.load(open(ij))['instruction'] if os.path.exists(ij) else '?'
    instr = instr.split('. Keep the shape')[0]
    row = [load_sq(pilot[sha]['front_view']), load_sq(f"{VIEWS_DIR}/{sha}.png")]
    row += [load_sq(r) for r in rend[:3]]
    S, BAN = 448, 46
    grid = Image.new('RGB', (S * len(row), S + BAN), (25, 25, 25))
    dr = ImageDraw.Draw(grid)
    dr.text((12, 11), f"{TAG} | {instr}   |   [before] [QIE edited view] [after x3]",
            fill=(255, 220, 120), font=font)
    for i, im in enumerate(row):
        grid.paste(im, (i * S, BAN))
    grid.save(f"{OUT}/{sha}.jpg", quality=90)
    n += 1
print(f"{n} grids -> {OUT}")
