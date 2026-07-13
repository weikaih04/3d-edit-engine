"""E1b part-local material edit — pure CPU UV-space texture merge.
Verified: null-after and edit-after glbs share IDENTICAL topology+UV (frozen geometry
makes TRELLIS.2 remesh/atlas deterministic), so:
  after_E1b = edit textures inside the P3-SAM part's UV region (feathered),
              null textures outside.
Part ids transfer from the ORIGINAL mesh to the TRELLIS mesh by nearest face centroid.
Usage: python e1b_merge.py [max_pairs]
"""
import os, sys, json, glob
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter
from scipy.spatial import cKDTree

B = "/fsx/hyperpod/weikaih_edit"
OUT = f"{B}/out_pairs/E1b"
EDIT_DIR = os.environ.get("E1B_EDIT_DIR", f"{B}/out_pilot/edit")
VIEWS_DIR = os.environ.get("E1B_VIEWS_DIR", f"{B}/edited_views")
NULL_DIR = os.environ.get("NULL_DIR", f"{B}/out_pilot/null")
OUT = os.environ.get("E1B_OUT", OUT)
# optional pre-chosen parts {sha: {pid: int}} — overrides random pick
PART_MAP = {}
if os.environ.get("E1B_PART_MAP") and os.path.exists(os.environ["E1B_PART_MAP"]):
    PART_MAP = json.load(open(os.environ["E1B_PART_MAP"]))
MAXP = int(sys.argv[1]) if len(sys.argv) > 1 else 10
rng = np.random.RandomState(11)
pilot = {a['sha']: a for a in json.load(open(f'{B}/pilot200.json'))}


def part_stats(fids):
    uniq, cnt = np.unique(fids, return_counts=True)
    share = cnt / cnt.sum()
    return [(u, c) for u, c, s in zip(uniq, cnt, share) if 0.04 <= s <= 0.4 and c > 80]


made = 0
for d in sorted(glob.glob(f"{EDIT_DIR}/*/")):
    if made >= MAXP:
        break
    sha = os.path.basename(d.rstrip('/'))
    nglb = f"{NULL_DIR}/{sha}/after.glb"
    eglb = d + "after.glb"
    pdir = f"{B}/out_p3sam/all/{sha}/"
    if not all(os.path.exists(p) for p in [nglb, eglb, pdir + 'face_ids.npy']):
        continue
    od = f"{OUT}/{sha}"
    if os.path.exists(f"{od}/meta.json"):
        made += 1
        continue
    try:
        fids = np.load(pdir + 'face_ids.npy')
        if sha in PART_MAP:
            pid = int(PART_MAP[sha]['pid'])
        else:
            parts = part_stats(fids)
            if not parts:
                continue
            pid = int(parts[rng.randint(len(parts))][0])

        orig = trimesh.load(pilot[sha]['glb'], force='mesh')
        # TRELLIS outputs live in the normalized [-0.5,0.5] cube; bring the
        # original mesh into the same frame before nearest-face transfer
        lo, hi = orig.vertices.min(0), orig.vertices.max(0)
        center, scale = (lo + hi) / 2, max((hi - lo).max(), 1e-9)
        tc_orig = (orig.triangles_center - center) / scale
        tree = cKDTree(tc_orig)

        nm = trimesh.load(nglb, force='mesh')
        em = trimesh.load(eglb, force='mesh')
        assert len(nm.faces) == len(em.faces)
        _, idx = tree.query(nm.triangles_center, k=1)
        hot_faces = fids[idx] == pid          # TRELLIS faces belonging to the part
        if hot_faces.mean() < 0.01:
            continue

        # rasterize part faces' UV triangles into an atlas mask
        uv = nm.visual.uv
        ntex = nm.visual.material.baseColorTexture
        W, H = ntex.size
        mask = Image.new('L', (W, H), 0)
        dr = ImageDraw.Draw(mask)
        for f in nm.faces[hot_faces]:
            tri = [(uv[v][0] * (W - 1), (1 - uv[v][1]) * (H - 1)) for v in f]
            dr.polygon(tri, fill=255)
        mask = mask.filter(ImageFilter.MaxFilter(5))       # dilate ~2px against seam bleed
        mask = mask.filter(ImageFilter.GaussianBlur(2))
        m = np.asarray(mask, dtype=np.float32)[..., None] / 255.0

        def merge(im_null, im_edit):
            if im_null is None or im_edit is None:
                return im_edit or im_null
            a = np.asarray(im_null.convert('RGBA'), dtype=np.float32)
            b = np.asarray(im_edit.convert('RGBA').resize(im_null.size), dtype=np.float32)
            out = a * (1 - m) + b * m
            return Image.fromarray(out.astype(np.uint8), 'RGBA')

        vis = em.visual.copy()
        matn, mate = nm.visual.material, em.visual.material
        vis.material.baseColorTexture = merge(matn.baseColorTexture, mate.baseColorTexture)
        if getattr(matn, 'metallicRoughnessTexture', None) is not None:
            vis.material.metallicRoughnessTexture = merge(
                matn.metallicRoughnessTexture, mate.metallicRoughnessTexture)
        merged = trimesh.Trimesh(vertices=em.vertices, faces=em.faces,
                                 visual=vis, process=False)
        os.makedirs(od, exist_ok=True)
        merged.export(f"{od}/after.glb")
        mask.save(f"{od}/uv_mask.png")
        instr = json.load(open(f"{VIEWS_DIR}/{sha}.json"))['instruction'].split('. Keep')[0]
        if instr.lower().startswith('change only'):
            final_instr = instr                     # v2: targeted instruction IS the pair instruction
            meta_extra = {"instruction": final_instr}
        else:
            mat = instr.replace("Change the object's material to ", "") \
                       .replace("Repaint the object in ", "").replace("Repaint the object with ", "")
            meta_extra = {"material": mat,
                          "instruction_template": "change the material of the {PART} to " + mat}
        json.dump({"sha": sha, "task": "E1b", "part_id": pid,
                   "part_face_share": float(hot_faces.mean()), **meta_extra},
                  open(f"{od}/meta.json", "w"), indent=1)
        # part_ref viz for naming
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        pts, fi = trimesh.sample.sample_surface(orig, 20000)
        hot = fids[fi] == pid
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
        axes[0].imshow(Image.open(pilot[sha]['front_view']).convert('RGB'))
        axes[0].set_title('asset'); axes[0].axis('off')
        for ax, (i, j), nmn in zip(axes[1:], [(0, 2), (1, 2), (0, 1)], ['front', 'side', 'top']):
            ax.scatter(pts[~hot][:, i], pts[~hot][:, j], c='lightgray', s=0.4)
            ax.scatter(pts[hot][:, i], pts[hot][:, j], c='red', s=0.6)
            ax.set_aspect('equal'); ax.axis('off'); ax.set_title(nmn)
        plt.tight_layout(); plt.savefig(f"{od}/part_ref.jpg", dpi=100); plt.close()
        made += 1
        print(f"[{sha[:8]}] E1b part={pid} share={hot_faces.mean():.2f}", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
print(f"E1B_DONE {made}")
