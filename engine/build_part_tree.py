"""Build a per-asset part-containment tree over P3-SAM parts.
A small part Q becomes a CHILD of a bigger part P if:
  - containment: >=55% of Q's surface samples have |generalized winding number| > 0.5
    w.r.t. P's face soup (robust to open shells — the eyes-inside-head case), OR
  - attachment: >=45% of Q's samples lie within eps of P's surface AND area(Q) <= 0.25*area(P)
Edit unit = part subtree (part + all descendants).
Output: out_p3sam/all/<sha>/part_tree.json {pid: {"parent": pid|-1, "rule": .., "score": ..}}
Usage: python build_part_tree.py [sha ...]   (default: all pilot assets with face_ids)
"""
import os, sys, json, glob
import numpy as np
import igl
from scipy.spatial import cKDTree
import trimesh
from edit_parts_lib import B, pilot, load_dump

INSIDE_FRAC, CONTACT_FRAC, SIZE_RATIO = 0.55, 0.45, 0.25
NSAMP = 40000


def build_tree(sha):
    meshes, fids = load_dump(sha)
    if meshes is None:
        return None
    merged = trimesh.util.concatenate([m.copy() for m in meshes])
    V = np.asarray(merged.vertices, dtype=np.float64)
    F = np.asarray(merged.faces, dtype=np.int64)
    areas = merged.area_faces
    diag = float(np.linalg.norm(np.ptp(V, axis=0)))
    np.random.seed(0)
    pts, fi = trimesh.sample.sample_surface(merged, NSAMP)
    ppid = fids[fi]
    fnrm = merged.face_normals[fi]

    uniq = np.unique(fids)
    info = {}
    for u in uniq:
        mask = ppid == u
        if mask.sum() < 30:
            continue
        p = pts[mask]
        info[int(u)] = {"area": float(areas[fids == u].sum()), "pts": p,
                        "nrm": fnrm[mask],
                        "lo": p.min(0), "hi": p.max(0)}

    ids = sorted(info, key=lambda u: -info[u]["area"])

    # ---- stage 1: mutual-contact clustering of similar-sized shells ----
    # semantic solids (a head, a torso) are often several nested/hugging shells of
    # comparable size; they must co-move. Merge P,Q into one cluster when the
    # smaller one hugs the bigger one (high contact fraction) or their bboxes
    # overlap heavily. A head touching the body only at a neck ring stays separate.
    parent_uf = {u: u for u in info}
    def find(u):
        while parent_uf[u] != u:
            parent_uf[u] = parent_uf[parent_uf[u]]
            u = parent_uf[u]
        return u
    def union(a, b):
        parent_uf[find(a)] = find(b)

    def bbox_iou(A, B):
        lo = np.maximum(A["lo"], B["lo"]); hi = np.minimum(A["hi"], B["hi"])
        inter = np.prod(np.maximum(hi - lo, 0))
        va = np.prod(A["hi"] - A["lo"]); vb = np.prod(B["hi"] - B["lo"])
        return inter / max(min(va, vb), 1e-12)          # relative to SMALLER box

    for i, p in enumerate(ids):
        for q in ids[i + 1:]:
            P, Q = info[p], info[q]
            if Q["area"] < SIZE_RATIO * P["area"]:
                continue                                # small ones handled by the tree
            biou = bbox_iou(P, Q)
            qs = Q["pts"][::max(1, len(Q["pts"]) // 600)]
            qn = Q["nrm"][::max(1, len(Q["nrm"]) // 600)]
            d, idx = cKDTree(P["pts"]).query(qs, k=1)
            near = d < 0.02 * diag
            contact = float(near.mean())
            if contact >= 0.35 or biou >= 0.50:
                union(p, q)                             # hugging / nested shells
                continue
            # seam tangent-continuity: two halves of ONE surface meet along a rim
            # where normals continue smoothly; a part sitting ON another meets at
            # a concave crease (normals disagree). Merge smooth-seam splits.
            if near.sum() >= 25:
                ang = np.degrees(np.arccos(np.clip(
                    np.abs((qn[near] * P["nrm"][idx[near]]).sum(1)), 0, 1)))
                if np.median(ang) < 22.0:
                    union(p, q)

    cluster_of = {u: find(u) for u in info}

    tree = {u: {"parent": -1, "rule": None, "score": 0.0} for u in info}
    for qi, q in enumerate(ids):
        Q = info[q]
        qs = Q["pts"][::max(1, len(Q["pts"]) // 800)]      # <=800 query pts
        best = None
        for p in ids[:qi]:                                  # only bigger parts
            P = info[p]
            if P["area"] < Q["area"] / SIZE_RATIO * SIZE_RATIO:  # p bigger by ordering anyway
                pass
            # bbox prefilter (Q center inside P bbox + 10% slack)
            c = (Q["lo"] + Q["hi"]) / 2
            slack = 0.10 * (P["hi"] - P["lo"] + 1e-9)
            if not np.all((c > P["lo"] - slack) & (c < P["hi"] + slack)):
                continue
            Fp = F[fids == p]
            w = igl.winding_number(V, np.asarray(Fp, dtype=np.int64),
                                   np.asarray(qs, dtype=np.float64))
            in_frac = float((np.abs(w) > 0.5).mean())
            if in_frac >= INSIDE_FRAC:
                cand = ("inside", in_frac, p)
            elif Q["area"] <= SIZE_RATIO * P["area"]:
                d, _ = cKDTree(P["pts"]).query(qs, k=1)
                cf = float((d < 0.02 * diag).mean())
                cand = ("attached", cf, p) if cf >= CONTACT_FRAC else None
            else:
                cand = None
            if cand and (best is None or cand[1] > best[1]):
                best = cand
        if best:
            tree[q] = {"parent": int(best[2]), "rule": best[0], "score": round(best[1], 3)}

    # ---- edit groups: the co-move set for every part ----
    # group(pid) = all parts in pid's cluster + every part whose tree-ancestor
    # chain lands in that cluster (children ride with their parents' cluster)
    kids = {}
    for c, v in tree.items():
        if v["parent"] != -1:
            kids.setdefault(cluster_of[v["parent"]], []).append(c)
    groups = {}
    for u in info:
        root = cluster_of[u]
        members = [x for x in info if cluster_of[x] == root]
        stack = list(members)
        seen = set(members)
        while stack:
            n = stack.pop()
            for ch in kids.get(cluster_of[n], []):
                if ch not in seen:
                    seen.add(ch); stack.append(ch)
            for ch, v in tree.items():
                if v["parent"] == n and ch not in seen:
                    seen.add(ch); stack.append(ch)
        groups[u] = sorted(int(x) for x in seen)

    return {"tree": {str(k): v for k, v in tree.items()},
            "cluster": {str(k): int(v) for k, v in cluster_of.items()},
            "group": {str(k): v for k, v in groups.items()}}


def subtree(tree, root):
    out, stack = [], [int(root)]
    kids = {}
    for c, v in tree.items():
        kids.setdefault(v["parent"], []).append(int(c))
    while stack:
        n = stack.pop()
        out.append(n)
        stack += kids.get(n, [])
    return out


if __name__ == "__main__":
    shas = sys.argv[1:] or [os.path.basename(d.rstrip('/'))
                            for d in sorted(glob.glob(f"{B}/out_p3sam/all/*/"))
                            if os.path.basename(d.rstrip('/')) in pilot()]
    for sha in shas:
        try:
            out = build_tree(sha)
            if out is None:
                continue
            op = f"{B}/out_p3sam/all/{sha}/part_tree.json"
            json.dump(out, open(op, "w"), indent=1)
            tree = out["tree"]
            nch = sum(1 for v in tree.values() if v["parent"] != -1)
            nin = sum(1 for v in tree.values() if v["rule"] == "inside")
            ncl = len(set(out["cluster"].values()))
            print(f"{sha[:8]}: {len(tree)} parts -> {ncl} clusters, {nch} children ({nin} inside)", flush=True)
        except Exception as e:
            print(f"{sha[:8]}: ERROR {str(e)[:80]}", flush=True)
    print("PART_TREE_DONE")
