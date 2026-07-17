"""E1b v3 prep: pick a SEMANTIC UNIT (VLM group) per E1a asset as the local-material
edit target. Emits e1b_v3_parts.json {sha: {name, pids, share, vis}} and targeted
instructions e1b_v3_instructions.json.
Usage: python e1b_v3_prep.py
"""
import os, json, glob
import numpy as np
from edit_parts_lib import B, pilot, load_dump, sample_with_parts, visibility, part_gate
from procedural_edits_v3 import semantic_units

MATS = ["polished gold metal", "red leather", "carved dark walnut wood",
        "white marble with gray veins", "brushed stainless steel",
        "glossy blue ceramic", "weathered rusty iron", "translucent green jade",
        "matte black carbon fiber", "bright red glossy paint"]

shas = sorted(json.load(open(f"{B}/e1a_instructions.json")).keys())
out, instr = {}, {}
i = 0
for sha in shas:
    if not part_gate(sha):
        continue
    units = semantic_units(sha)
    if not units:
        continue
    meshes, fids = load_dump(sha)
    if meshes is None:
        continue
    areas = np.concatenate([m.area_faces for m in meshes])
    tot = areas.sum()
    pts, ppid = sample_with_parts(meshes, fids)
    best = None
    for name, pids in units:
        share = float(areas[np.isin(fids, list(pids))].sum() / tot)
        if not (0.05 <= share <= 0.5):
            continue
        upid = np.where(np.isin(ppid, list(pids)), 1, 0)
        vis = visibility(pts, upid, 1)
        if vis < 0.35:
            continue
        score = vis * share
        if best is None or score > best[0]:
            best = (score, name, sorted(pids), share, vis)
    if best is None:
        continue
    _, name, pids, share, vis = best
    out[sha] = {"name": name, "pids": pids, "share": round(share, 3), "vis": round(vis, 2)}
    mat = MATS[i % len(MATS)]
    instr[sha] = (f"Change only the {name} to {mat}. Keep every other part and the "
                  f"shape, pose, silhouette and background exactly the same.")
    i += 1
    print(sha[:8], name, f"share={share:.2f} vis={vis:.2f} | {mat}", flush=True)

json.dump(out, open(f"{B}/e1b_v3_parts.json", "w"), indent=1)
json.dump(instr, open(f"{B}/e1b_v3_instructions.json", "w"), indent=1)
print(f"E1B_V3_PREP_DONE {len(out)}")
