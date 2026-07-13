"""Reverse pairs — free doubled data with naturally-correct semantics:
  E2  = inverse of round-2 E3 removal:   (E3.after, "add a {PART}", E3.before)
  E6r = inverse of round-2 E6 transform: (E6.after, inverse instruction, E6.before)
  E1a-rev: (edit_v2 after, "restore original look: <texture caption appearance>", null_v2 after)
Outputs into out_pairs2/{E2,E6,E1a_rev}. part_ref carried over so naming reuses it.
Usage: python reverse_pairs.py
"""
import os, json, glob, shutil

B = "/fsx/hyperpod/weikaih_edit"
OUT = f"{B}/out_pairs2"

# --- E2 from E3 inverse
n = 0
for mp in sorted(glob.glob(f"{OUT}/E3/*/meta.json")):
    d = os.path.dirname(mp)
    m = json.load(open(mp))
    od = f"{OUT}/E2/{m['sha']}__addback_p{m['part_id']}"
    os.makedirs(od, exist_ok=True)
    # reversed roles: before = removed state, after = full state
    for src, dst in [(f"{d}/after.glb", f"{od}/before.glb"),
                     (f"{d}/before.glb", f"{od}/after.glb"),
                     (f"{d}/part_ref.jpg", f"{od}/part_ref.jpg")]:
        if not os.path.exists(dst):
            os.link(src, dst)
    json.dump({"sha": m['sha'], "task": "E2", "part_id": m['part_id'],
               "source": "inverse_E3", "area_share": m.get('area_share'),
               "vis": m.get('vis'),
               "instruction_template": "add a {PART} to the object"},
              open(f"{od}/meta.json", "w"), indent=1)
    n += 1
print(f"E2 (inverse E3): {n}")

# --- E6 reverse
n = 0
for mp in sorted(glob.glob(f"{OUT}/E6/*/meta.json")):
    d = os.path.dirname(mp)
    m = json.load(open(mp))
    if 'inverse_template' not in m or m.get('source') == 'inverse_E6':
        continue
    od = f"{OUT}/E6/{m['sha']}__p{m['part_id']}_{m['op']}_rev"
    os.makedirs(od, exist_ok=True)
    for src, dst in [(f"{d}/after.glb", f"{od}/before.glb"),
                     (f"{d}/before.glb", f"{od}/after.glb"),
                     (f"{d}/part_ref.jpg", f"{od}/part_ref.jpg")]:
        if not os.path.exists(dst):
            os.link(src, dst)
    json.dump({"sha": m['sha'], "task": "E6", "part_id": m['part_id'],
               "op": m['op'] + "_rev", "source": "inverse_E6",
               "area_share": m.get('area_share'), "vis": m.get('vis'),
               "instruction_template": m['inverse_template']},
              open(f"{od}/meta.json", "w"), indent=1)
    n += 1
print(f"E6 reverse: {n}")

# --- E1a reverse (restore original appearance)
caps = {}
for r in json.load(open('/fsx/hyperpod/weikaih_cap/FINAL_sfv1_texture_captions.json'))['results']:
    caps[r['sha']] = r
n = 0
for mp in sorted(glob.glob(f"{B}/out_pilot/edit_v2/*/meta.json")):
    m = json.load(open(mp))
    if m.get('error'):
        continue
    sha = m['sha']
    e_glb = f"{B}/out_pilot/edit_v2/{sha}/after.glb"
    n_glb = f"{B}/out_pilot/null_v2/{sha}/after.glb"
    if not (os.path.exists(e_glb) and os.path.exists(n_glb) and sha in caps):
        continue
    try:
        raw = caps[sha]['raw']
        js = raw[raw.index('{'):raw.rindex('}') + 1]
        app = json.loads(js).get('appearance', '').strip()
    except Exception:
        app = ''
    if not app:
        continue
    od = f"{OUT}/E1a_rev/{sha}"
    os.makedirs(od, exist_ok=True)
    for src, dst in [(e_glb, f"{od}/before.glb"), (n_glb, f"{od}/after.glb")]:
        if not os.path.exists(dst):
            os.link(src, dst)
    json.dump({"sha": sha, "task": "E1a_rev", "source": "inverse_E1a",
               "forward_instruction": json.load(open(f"{B}/edited_views_v2/{sha}.json"))['instruction'].split('. Keep')[0],
               "instruction": f"restore the object's original appearance: {app}"},
              open(f"{od}/meta.json", "w"), indent=1)
    n += 1
print(f"E1a reverse: {n}")
print("REVERSE_PAIRS_DONE")
