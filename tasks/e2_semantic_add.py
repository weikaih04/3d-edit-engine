"""E2 v2: 'add a real semantic part' pairs via reverse-of-removal on ADD-ABLE parts.
Selects semantic units whose NAME is an additive accessory (hat, wings, tail, ...),
removes it, and emits the reverse pair: (without-part, "add a <part>", with-part).
Unlike E2=E3-inverse-on-random-parts, the instruction is a natural add and the
after-part is a real, well-placed accessory.
Usage: python e2_semantic_add.py [n]
Output: out_pairs3/E2sem/<sha>__add_<name>/{before,after}.glb + meta
"""
import os, sys, json, glob
import numpy as np
from edit_parts_lib import (B, pilot, load_dump, export_scene, apply_face_mask,
                            sample_with_parts, visibility, part_gate)
from procedural_edits_v3 import semantic_units

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
OUT = f"{B}/out_pairs3/E2sem"

# additive accessories a user would say "add a ___" — sit ON the object, removable
ADDABLE = {
    'hat', 'cap', 'helmet', 'hair', 'head wrap', 'crown', 'mask',
    'wings', 'wing', 'tail', 'horn', 'horns', 'antenna', 'antennae', 'ears',
    'backpack', 'bag', 'cape', 'cloak', 'scarf',
    'weapon', 'sword', 'blade', 'staff', 'shield', 'gun',
    'saddle', 'reins', 'belt', 'glasses', 'spectacles',
    'boots', 'gloves', 'shoes', 'spoiler', 'flag', 'sail',
    'fins', 'spikes', 'petals', 'gem', 'jewel', 'bow', 'ribbon',
}
# main-body / structural names that must NOT be "added"
BLOCK = {'body', 'torso', 'base', 'main body', 'upper body', 'legs', 'arms', 'limbs',
         'head', 'main', 'frame', 'walls', 'seat', 'body and cape', 'legs and feet',
         'base platform', 'roof structure', 'bottle body', 'box body', 'abdomen',
         'thorax', 'main frame', 'feet', 'hands', 'central body'}


def is_addable(name):
    n = name.strip().lower()
    if n in BLOCK:
        return False
    if n in ADDABLE:
        return True
    return any(tok in ADDABLE for tok in n.split())


def face_slices(meshes):
    out, off = [], 0
    for m in meshes:
        out.append((off, off + len(m.faces))); off += len(m.faces)
    return out


shas = [p.split('/')[-2] for p in sorted(glob.glob(f'{B}/vlm_merge/*/groups.json'))]
made = 0
for sha in shas:
    if made >= N:
        break
    if sha not in pilot():
        continue
    units = semantic_units(sha)
    if not units:
        continue
    add_units = [(nm, pids) for nm, pids in units if is_addable(nm)]
    if not add_units:
        continue
    meshes, fids = load_dump(sha)
    if meshes is None:
        continue
    areas = np.concatenate([m.area_faces for m in meshes]); tot = areas.sum()
    pts, ppid = sample_with_parts(meshes, fids)
    for nm, pids in add_units:
        share = float(areas[np.isin(fids, list(pids))].sum() / tot)
        if not (0.01 <= share <= 0.45):
            continue
        vis = visibility(pts, np.where(np.isin(ppid, list(pids)), 1, 0), 1)
        if vis < 0.35:
            continue
        without = apply_face_mask(meshes, ~np.isin(fids, list(pids)))
        if not without:
            continue
        od = f"{OUT}/{sha}__add_{nm.replace(' ', '_')}"
        os.makedirs(od, exist_ok=True)
        # before = without the part, after = full (original)
        export_scene(without, f"{od}/before.glb")
        export_scene(meshes, f"{od}/after.glb")
        json.dump({"sha": sha, "task": "E2", "source": "semantic_add",
                   "unit": {"name": nm, "pids": sorted(int(p) for p in pids),
                            "share": round(share, 3), "vis": round(vis, 2)},
                   "instruction": f"add a {nm} to the object"},
                  open(f"{od}/meta.json", "w"), indent=1)
        made += 1
        print(f"[{sha[:8]}] add '{nm}' share={share:.2f} vis={vis:.2f}", flush=True)
        break   # one add-part per asset for diversity
print(f"E2_SEMANTIC_DONE {made}")
