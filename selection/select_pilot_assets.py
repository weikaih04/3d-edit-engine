"""Select 200 pilot assets: judge yes ∩ SketchfabV1 ∩ raw glb on disk,
stratified over part_complexity x color_richness so every edit type has fodder.
Output: pilot200.json [{sha, glb, front_view, part_complexity, color_richness, ...}]
"""
import json, os, random

JUDGE = "/fsx/hyperpod/weikaih_cap/judge_scores.jsonl"
GLBS = "/fsx/sfr/weikaih/3dgen/data/trellis2/SketchfabV1/raw/glbs"
REND = "/fsx/sfr/weikaih/3dgen/data/trellis2/SketchfabV1/renders_cond"
OUT = "/fsx/hyperpod/weikaih_edit/pilot200.json"
random.seed(0)

cands = []
for line in open(JUDGE):
    r = json.loads(line)
    if r.get("subset") != "SketchfabV1" or r.get("recommended") != "yes":
        continue
    if r.get("validity") != "single_object":
        continue
    cands.append(r)
print(f"yes+single_object SketchfabV1: {len(cands):,}")

# stratify: 4 buckets
buckets = {"complex_rich": [], "complex_plain": [], "simple_rich": [], "simple_plain": []}
for r in cands:
    pc, cr = r.get("part_complexity", 0), r.get("color_richness", 0)
    key = ("complex" if pc >= 4 else "simple") + "_" + ("rich" if cr >= 4 else "plain")
    buckets[key].append(r)
for k, v in buckets.items():
    print(k, len(v))

quota = {"complex_rich": 70, "complex_plain": 50, "simple_rich": 50, "simple_plain": 30}
picked = []
for k, n in quota.items():
    pool = buckets[k]
    random.shuffle(pool)
    got = 0
    for r in pool:
        sha = r["sha"]
        glb = f"{GLBS}/{sha}.glb"
        rd = f"{REND}/{sha}"
        if not os.path.exists(glb):
            continue
        if not os.path.isdir(rd):
            continue
        views = sorted(os.listdir(rd))
        if not views:
            continue
        picked.append({
            "sha": sha, "glb": glb,
            "renders_dir": rd,
            "front_view": f"{rd}/{views[0]}",
            "part_complexity": r.get("part_complexity"),
            "color_richness": r.get("color_richness"),
            "structural_score": r.get("structural_score"),
            "texture_score": r.get("texture_score"),
            "category": r.get("category"), "style": r.get("style"),
            "bucket": k,
        })
        got += 1
        if got >= n:
            break
    print(f"{k}: picked {got}/{n}")

json.dump(picked, open(OUT, "w"), indent=1)
print(f"TOTAL {len(picked)} -> {OUT}")
