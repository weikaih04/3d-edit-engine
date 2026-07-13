"""Stack QC grids into contact sheets (6 rows/sheet) for claude verification.
Each row keeps its instruction banner; a left margin shows pair key for verdict keying.
Usage: python make_sheets.py  -> qc_sheets/sheet_###.jpg + qc_sheets/index.json
"""
import os, glob, json
from PIL import Image, ImageDraw, ImageFont

B = "/fsx/hyperpod/weikaih_edit"
OUT = f"{B}/qc_sheets"
os.makedirs(OUT, exist_ok=True)
for f in glob.glob(f"{OUT}/*"):
    os.remove(f)

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
except Exception:
    font = ImageFont.load_default()

items = []  # (key, path)
for g in sorted(glob.glob(f"{B}/out_pairs/compare/*.jpg")):
    name = os.path.basename(g)[:-4]
    task, pname = name.split('__', 1)
    items.append((f"{task}/{pname}", g))
for task, cmp_dir, edir in [('E1a', 'compare_e1a_v2', 'out_pilot/edit_v2'),
                            ('E1c', 'compare_e1c', 'out_pilot/e1c'),
                            ('E5', 'compare_e5', 'out_pilot/e5')]:
    for g in sorted(glob.glob(f"{B}/out_pilot/{cmp_dir}/*.jpg")):
        sha = os.path.basename(g)[:-4]
        items.append((f"{task}/{sha}", g))

W = 1750
MARG = 210
rows_per = 6
sheets, idx = [], {}
for i in range(0, len(items), rows_per):
    chunk = items[i:i + rows_per]
    ims = []
    for key, p in chunk:
        im = Image.open(p)
        s = (W - MARG) / im.width
        ims.append((key, im.resize((W - MARG, int(im.height * s)), Image.LANCZOS)))
    H = sum(im.height for _, im in ims)
    sheet = Image.new('RGB', (W, H), (10, 10, 10))
    dr = ImageDraw.Draw(sheet)
    y = 0
    sid = len(sheets)
    for j, (key, im) in enumerate(ims):
        sheet.paste(im, (MARG, y))
        rid = f"S{sid:02d}R{j}"
        dr.text((8, y + 8), rid, fill=(120, 255, 120), font=font)
        for k, seg in enumerate([key.split('/')[0]] + [key.split('/')[1][m:m+22] for m in range(0, min(len(key.split('/')[1]), 44), 22)]):
            dr.text((8, y + 30 + 20 * k), seg, fill=(200, 200, 120), font=font)
        idx[rid] = key
        y += im.height
    sp = f"{OUT}/sheet_{sid:03d}.jpg"
    sheet.save(sp, quality=85)
    sheets.append(sp)
json.dump(idx, open(f"{OUT}/index.json", "w"), indent=1)
print(f"{len(items)} grids -> {len(sheets)} sheets")
print("SHEETS_DONE")
