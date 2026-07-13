"""E10 v2: compose multi-turn chains ONLY from gate-passed round-2 pairs
(the V1-plan design: assembly of accepted pairs, no new generation).
Chain = two gated edits on the SAME asset with DIFFERENT parts:
  state0 = before, state1 = after(edit1), state2 = apply edit2's delta on state1.
Supported combos: E3(remove p1) + E6(transform p2)  [both procedural, replayable]
Usage: python e10_v2_compose.py
"""
import os, json, glob, shutil
import numpy as np
import trimesh
from edit_parts_lib import B, load_dump, face_slices, export_scene, save_part_ref, sample_with_parts

OUT = f"{B}/out_pairs2/E10"
judge = json.load(open(f"{B}/judge_round2_results.json"))
passed = {k for k, v in judge.items() if v.get('match') == 'yes'}

# collect gated E3 and E6 (forward only) pairs per sha
e3 = {}
for mp in glob.glob(f"{B}/out_pairs2/E3/*/meta.json"):
    m = json.load(open(mp))
    key = f"E3/{os.path.basename(os.path.dirname(mp))}"
    if key in passed:
        e3[m['sha']] = m
e6 = {}
for mp in glob.glob(f"{B}/out_pairs2/E6/*/meta.json"):
    m = json.load(open(mp))
    if m.get('source') == 'inverse_E6':
        continue
    key = f"E6/{os.path.basename(os.path.dirname(mp))}"
    if key in passed:
        e6.setdefault(m['sha'], m)

shared = [s for s in e3 if s in e6 and e3[s]['part_id'] != e6[s]['part_id']]
print(f"gated E3∩E6 same-asset different-part: {len(shared)}")

shutil.rmtree(OUT, ignore_errors=True)
made = 0
for sha in shared:
    m3, m6 = e3[sha], e6[sha]
    meshes, fids = load_dump(sha)
    if meshes is None:
        continue
    # state1: remove E3 part
    keep = fids != m3['part_id']
    s1, kept = [], []
    for m, (a, b) in zip(meshes, face_slices(meshes)):
        mm = m.copy(); k = keep[a:b]
        if not k.any():
            continue
        if not k.all():
            mm.update_faces(k); mm.remove_unreferenced_vertices()
        if len(mm.faces):
            s1.append(mm); kept.append(fids[a:b][k])
    f1 = np.concatenate(kept)
    if (f1 == m6['part_id']).sum() < 40:
        continue
    # state2: replay E6 op on its part
    tc = np.concatenate([m.triangles_center for m in s1])
    c = tc[f1 == m6['part_id']].mean(axis=0)
    op = m6['op']
    T = np.eye(4)
    if op == 'scale':
        T[:3, :3] *= 1.5; T[:3, 3] = c - 1.5 * c
    elif op == 'rotate':
        T = trimesh.transformations.rotation_matrix(np.deg2rad(35), [0, 0, 1], c)
    else:
        ext = np.ptp(np.concatenate([m.vertices for m in meshes]), axis=0)
        ax = int(np.argmax(ext[:2])); off = np.zeros(3); off[ax] = 0.3 * ext[ax]
        T[:3, 3] = off
    s2 = []
    for m, (a, b) in zip(s1, face_slices(s1)):
        mm = m.copy()
        pf = np.where(f1[a:b] == m6['part_id'])[0]
        if len(pf):
            vids = np.unique(mm.faces[pf])
            v = mm.vertices[vids]
            mm.vertices[vids] = (np.c_[v, np.ones(len(v))] @ T.T)[:, :3]
        s2.append(mm)
    od = f"{OUT}/{sha}"
    os.makedirs(od, exist_ok=True)
    export_scene(meshes, f"{od}/state0.glb")
    export_scene(s1, f"{od}/state1.glb")
    export_scene(s2, f"{od}/state2.glb")
    cache = sample_with_parts(meshes, fids)
    save_part_ref(meshes, fids, m3['part_id'], sha, f"{od}/part1_ref.jpg", cache)
    save_part_ref(meshes, fids, m6['part_id'], sha, f"{od}/part2_ref.jpg", cache)
    json.dump({"sha": sha, "task": "E10", "source": "composed_from_gated",
               "turns": [
                   {"part_id": m3['part_id'],
                    "instruction": m3.get('instruction', m3['instruction_template'])},
                   {"part_id": m6['part_id'],
                    "instruction": m6.get('instruction', m6['instruction_template'])}]},
              open(f"{od}/meta.json", "w"), indent=1)
    made += 1
    print(sha[:8], m3.get('instruction'), '->', m6.get('instruction'), flush=True)
print(f"E10_V2_DONE {made}")
