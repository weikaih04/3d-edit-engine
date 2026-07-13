"""Build dual-blind judge manifests.
Usage: python build_judge_manifest.py round1|round2 <out.json>
round1: the 125 verified pairs (grids in out_pairs/compare + out_pilot/compare_*) — calibration.
round2: out_pairs2 pairs (grids in out_pairs2/compare).
"""
import os, sys, json, glob

B = "/fsx/hyperpod/weikaih_edit"
mode, outp = sys.argv[1], sys.argv[2]
items = []


def instr_of(meta):
    if 'turns' in meta:
        return " THEN ".join((t.get('instruction') or t.get('instruction_template', '?')) for t in meta['turns'])
    return meta.get('instruction') or meta.get('instruction_template', '?')


if mode == 'round1':
    for mp in sorted(glob.glob(f"{B}/out_pairs/*/*/meta.json")):
        d = os.path.dirname(mp)
        task, name = d.split('/')[-2], os.path.basename(d)
        g = f"{B}/out_pairs/compare/{task}__{name}.jpg"
        items.append({"key": f"{task}/{name}", "grid": g,
                      "instruction": instr_of(json.load(open(mp)))})
    for task, cd, ed, vd in [('E1a', 'compare_e1a_v2', 'edit_v2', 'edited_views_v2'),
                             ('E1c', 'compare_e1c', 'e1c', 'e1c_views'),
                             ('E5', 'compare_e5', 'e5', 'e5_views')]:
        for mp in sorted(glob.glob(f"{B}/out_pilot/{ed}/*/meta.json")):
            m = json.load(open(mp))
            if m.get('error'):
                continue
            sha = m['sha']
            ij = f"{B}/{vd}/{sha}.json"
            instr = json.load(open(ij))['instruction'].split('. Keep')[0] if os.path.exists(ij) else '?'
            items.append({"key": f"{task}/{sha}",
                          "grid": f"{B}/out_pilot/{cd}/{sha}.jpg", "instruction": instr})
else:
    for mp in sorted(glob.glob(f"{B}/out_pairs2/*/*/meta.json")):
        d = os.path.dirname(mp)
        task, name = d.split('/')[-2], os.path.basename(d)
        g = f"{B}/out_pairs2/compare/{task}__{name}.jpg"
        items.append({"key": f"{task}/{name}", "grid": g,
                      "instruction": instr_of(json.load(open(mp)))})

json.dump(items, open(outp, 'w'), indent=1)
print(f"{mode}: {len(items)} items -> {outp}")
