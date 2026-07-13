"""QC grids for out_pairs: per pair one row image with instruction banner.
Procedural (E2/E3/E4/E6/E7/E8/E9): [before r0][before r1][after r0][after r1]
E1b: [orig front_view][edited cond view][after r0][after r1]
E10: [state0 r0][state1 r0][state2 r0]
Output: out_pairs/compare/<task>__<name>.jpg
Usage: python grid_pairs.py
"""
import os, glob, json
from PIL import Image, ImageDraw, ImageFont

B = "/fsx/hyperpod/weikaih_edit"
ROOT = os.environ.get("PAIRS_ROOT", f"{B}/out_pairs")
OUT = f"{ROOT}/compare"
os.makedirs(OUT, exist_ok=True)
pilot = {a['sha']: a for a in json.load(open(f'{B}/pilot200.json'))}

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
except Exception:
    font = ImageFont.load_default()

def sq(p, size=420):
    im = Image.open(p)
    if im.mode == 'RGBA':
        bg = Image.new('RGBA', im.size, (40, 40, 40, 255))
        im = Image.alpha_composite(bg, im)
    return im.convert('RGB').resize((size, size), Image.LANCZOS)

n = 0
for d in sorted(glob.glob(f"{ROOT}/*/*/")):
    task = d.split('/')[-3]
    name = os.path.basename(d.rstrip('/'))
    if task == 'compare':
        continue
    mp = f"{d}meta.json"
    if not os.path.exists(mp):
        continue
    meta = json.load(open(mp))
    if task == 'E10':
        instr = " THEN ".join((t.get('instruction') or t.get('instruction_template', '?'))
                              for t in meta['turns'])
        paths = [f"{d}state{i}_r000.webp" for i in range(3)]
    elif task == 'E1b':
        instr = meta.get('instruction') or meta.get('instruction_template', '?')
        paths = [pilot[meta['sha']]['front_view'],
                 f"{d}after_r000.webp", f"{d}after_r001.webp"]
    else:
        instr = meta.get('instruction') or meta.get('instruction_template', '?')
        paths = [f"{d}before_r000.webp", f"{d}before_r001.webp",
                 f"{d}after_r000.webp", f"{d}after_r001.webp"]
        # part-closeup panels when available (look-at-part renders)
        if os.path.exists(f"{d}before_z000.webp") and os.path.exists(f"{d}after_z000.webp"):
            paths += [f"{d}before_z000.webp", f"{d}after_z000.webp"]
    if not all(os.path.exists(p) for p in paths):
        continue
    row = [sq(p) for p in paths]
    S, BAN = 420, 44
    grid = Image.new('RGB', (S * len(row), S + BAN), (25, 25, 25))
    dr = ImageDraw.Draw(grid)
    dr.text((10, 10), f"{task} | {instr}", fill=(255, 220, 120), font=font)
    for i, im in enumerate(row):
        grid.paste(im, (i * S, BAN))
    grid.save(f"{OUT}/{task}__{name}.jpg", quality=88)
    n += 1
print(f"{n} grids -> {OUT}")
print("GRID_PAIRS_DONE")
