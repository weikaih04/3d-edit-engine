"""Deterministic texture transfer (rebake): copy the ORIGINAL asset's texture onto a
TRELLIS-topology mesh's UV atlas. Kills the text-misalignment problem: regions that an
edit leaves unchanged keep the EXACT original texels instead of a generative re-synthesis.

Method: densely sample the TARGET surface (points + their UV + atlas texel coords),
look up each point's color from the ORIGINAL mesh via nearest sampled original point
(KDTree over 4M original surface samples colored through the original UV/texture),
splat colors into the target atlas, hole-fill by dilation.

API: rebake(original_glb_path, target_mesh_or_path, texture_size=1024) -> PIL.Image (RGBA)
CLI: python rebake_texture.py <original.glb> <target.glb> <out_textured.glb>
"""
import sys
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree


def _sample_colored(mesh, n):
    """surface samples + per-sample RGBA color read through the mesh's own UV/texture"""
    np.random.seed(0)
    pts, fi = trimesh.sample.sample_surface(mesh, n)
    uv = getattr(mesh.visual, 'uv', None)
    mat = getattr(mesh.visual, 'material', None)
    tex = getattr(mat, 'baseColorTexture', None) if mat is not None else None
    if uv is None or tex is None:
        cols = np.full((len(pts), 4), 200, np.uint8)
        return pts, cols
    tri = mesh.faces[fi]
    bary = trimesh.triangles.points_to_barycentric(mesh.triangles[fi], pts)
    uvp = (uv[tri] * bary[..., None]).sum(axis=1)
    T = np.asarray(tex.convert('RGBA'))
    H, W = T.shape[:2]
    x = np.clip((uvp[:, 0] % 1.0) * (W - 1), 0, W - 1).astype(int)
    y = np.clip((1 - uvp[:, 1] % 1.0) * (H - 1), 0, H - 1).astype(int)
    return pts, T[y, x]


def _norm_frame(v):
    lo, hi = v.min(0), v.max(0)
    c = (lo + hi) / 2
    s = max((hi - lo).max(), 1e-9)
    return c, s


def rebake(original, target, texture_size=1024, n_src=4_000_000, n_tgt=4_000_000):
    if isinstance(original, str):
        original = trimesh.load(original, force='mesh')
    if isinstance(target, str):
        target = trimesh.load(target, force='mesh')
    # bring both into the same normalized frame (TRELLIS lives in [-0.5,0.5])
    oc, osc = _norm_frame(original.vertices)
    tc, tsc = _norm_frame(target.vertices)

    src_pts, src_col = _sample_colored(original, n_src)
    src_pts = (src_pts - oc) / osc
    tree = cKDTree(src_pts)

    np.random.seed(1)
    tpts, tfi = trimesh.sample.sample_surface(target, n_tgt)
    tuv = target.visual.uv
    tri = target.faces[tfi]
    bary = trimesh.triangles.points_to_barycentric(target.triangles[tfi], tpts)
    uvp = (tuv[tri] * bary[..., None]).sum(axis=1)
    tpts_n = ((tpts - tc) / tsc)

    _, idx = tree.query(tpts_n, k=1, workers=-1)
    cols = src_col[idx].astype(np.float32)

    S = texture_size
    x = np.clip((uvp[:, 0] % 1.0) * (S - 1), 0, S - 1).astype(int)
    y = np.clip((1 - uvp[:, 1] % 1.0) * (S - 1), 0, S - 1).astype(int)
    acc = np.zeros((S, S, 4), np.float64)
    cnt = np.zeros((S, S), np.int64)
    np.add.at(acc, (y, x), cols)
    np.add.at(cnt, (y, x), 1)
    filled = cnt > 0
    atlas = np.zeros((S, S, 4), np.uint8)
    atlas[filled] = (acc[filled] / cnt[filled, None]).astype(np.uint8)
    # hole fill: dilate up to 8 rounds
    for _ in range(8):
        holes = ~filled
        if not holes.any():
            break
        from scipy.ndimage import binary_dilation, distance_transform_edt
        ind = distance_transform_edt(holes, return_distances=False, return_indices=True)
        atlas[holes] = atlas[ind[0][holes], ind[1][holes]]
        break
    atlas[..., 3] = 255
    return Image.fromarray(atlas, 'RGBA')


if __name__ == "__main__":
    orig_p, tgt_p, out_p = sys.argv[1], sys.argv[2], sys.argv[3]
    tgt = trimesh.load(tgt_p, force='mesh')
    img = rebake(orig_p, tgt)
    vis = tgt.visual.copy()
    vis.material.baseColorTexture = img
    out = trimesh.Trimesh(vertices=tgt.vertices, faces=tgt.faces, visual=vis, process=False)
    out.export(out_p)
    print(f"REBAKE_DONE -> {out_p}")
