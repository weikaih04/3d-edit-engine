"""Visualize P3-SAM segmentation: sample surface points, color by part id,
scatter from 3 orthographic views + original render. One jpg grid per asset.
Usage: python viz_p3sam.py [n_assets]
"""
import os, sys, glob, json
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
OUT = "/fsx/hyperpod/weikaih_edit/out_p3sam/viz"
os.makedirs(OUT, exist_ok=True)
pilot = {a['sha']: a for a in json.load(open('/fsx/hyperpod/weikaih_edit/pilot200.json'))}

done = [d for d in sorted(glob.glob('/fsx/hyperpod/weikaih_edit/out_p3sam/all/*/'))
        if os.path.exists(d + 'aabb.npy')]
# prefer part-rich assets for the demo
def npart(d):
    return len(np.unique(np.load(d + 'face_ids.npy')))
done = sorted(done, key=npart, reverse=True)[:N]

cmap = plt.get_cmap('tab20')
for d in done:
    sha = os.path.basename(d.rstrip('/'))
    mesh = trimesh.load(d + 'mesh_clean.glb', force='mesh')
    fids = np.load(d + 'face_ids.npy')
    pts, fidx = trimesh.sample.sample_surface(mesh, 30000)
    pid = fids[fidx]
    uniq = np.unique(pid)
    colors = {u: cmap(i % 20) for i, u in enumerate(uniq)}
    c = np.array([colors[p] for p in pid])

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    # original render
    if sha in pilot:
        axes[0].imshow(Image.open(pilot[sha]['front_view']).convert('RGB'))
    axes[0].set_title('original'); axes[0].axis('off')
    views = [(0, 1), (0, 2), (1, 2)]
    names = ['front(XY)', 'side(XZ)', 'top(YZ)']
    for ax, (i, j), nm in zip(axes[1:], views, names):
        ax.scatter(pts[:, i], pts[:, j], c=c, s=0.5)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(f'{nm} · {len(uniq)} parts')
    plt.tight_layout()
    plt.savefig(f'{OUT}/{sha}.jpg', dpi=110)
    plt.close()
    print(sha, len(uniq), 'parts')
print('viz ->', OUT)
