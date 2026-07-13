"""E1b v2 prep (CPU): for the 30 E1a assets pick the most visible usable part
(area-share + z-buffer visibility), save refs for naming + the part map.
Output: e1b_v2_parts.json {sha: {pid, area_share, vis}}, e1b_v2_refs/<sha>.jpg
"""
import os, json, glob
import numpy as np
from edit_parts_lib import B, pilot, load_dump, part_stats_v2, save_part_ref, part_gate

REFS = f"{B}/e1b_v2_refs"
os.makedirs(REFS, exist_ok=True)
shas = sorted(json.load(open(f"{B}/e1a_instructions.json")).keys())

out = {}
for sha in shas:
    if not part_gate(sha):
        continue
    meshes, fids = load_dump(sha)
    if meshes is None:
        continue
    parts, cache = part_stats_v2(meshes, fids, min_share=0.05, max_share=0.40)
    if not parts:
        continue
    best = max(parts, key=lambda p: p['vis'] * p['area_share'])
    out[sha] = best
    save_part_ref(meshes, fids, best['pid'], sha, f"{REFS}/{sha}.jpg", cache)
    print(sha[:8], best, flush=True)

json.dump(out, open(f"{B}/e1b_v2_parts.json", "w"), indent=1)
print(f"E1B_V2_PREP_DONE {len(out)} assets")
