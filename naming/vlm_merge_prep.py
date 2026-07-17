"""VLM semantic-merge prototype, step 1 (CPU):
Color-code each GEOMETRY CLUSTER of an asset with a distinct flat color, render
3 views with blender, compose one image with a color legend for the VLM.
Output: vlm_merge/<sha>/{colored.glb, view_*.webp, panel.jpg, colors.json}
Usage: python vlm_merge_prep.py <sha> [sha ...]
"""
import os, sys, json, glob, subprocess, math
import numpy as np
import trimesh
from edit_parts_lib import B, pilot, load_dump, face_slices

BLENDER = "/tmp/blender-4.5.1-linux-x64/blender"
SCRIPT = "/fsx/sfr/weikaih/trellis2-data-pipeline/toolkit/blender_script/render_cond.py"
_R = 2.0; _FOV = 2 * math.asin(math.sqrt(3) / 2 / _R)
VIEWS = [{"yaw": y, "pitch": 0.3, "radius": _R, "fov": _FOV} for y in (0.6, 2.2, 3.8)]
PALETTE = [("red", (220, 40, 40)), ("blue", (40, 80, 230)), ("green", (40, 180, 60)),
           ("yellow", (235, 200, 30)), ("purple", (150, 60, 200)), ("orange", (240, 130, 30)),
           ("cyan", (40, 200, 210)), ("magenta", (230, 60, 180)), ("brown", (140, 90, 40)),
           ("pink", (250, 170, 190)), ("olive", (128, 128, 0)), ("navy", (20, 30, 120))]


def prep(sha):
    od = f"{B}/vlm_merge/{sha}"
    os.makedirs(od, exist_ok=True)
    meshes, fids = load_dump(sha)
    data = json.load(open(f"{B}/out_p3sam/all/{sha}/part_tree.json"))
    cluster = {int(k): v for k, v in data['cluster'].items()}
    tree = {int(k): v for k, v in data['tree'].items()}
    # attach contained/attached children to their parent's cluster for coloring
    eff = {}
    for pid, cl in cluster.items():
        eff[pid] = cl
    for pid, v in tree.items():
        if v['parent'] != -1 and cluster.get(pid) == pid:  # child in own cluster
            eff[pid] = cluster.get(v['parent'], pid)
    merged = trimesh.util.concatenate([m.copy() for m in meshes])
    areas = merged.area_faces
    # rank effective clusters by area
    ca = {}
    for pid, cl in eff.items():
        ca[cl] = ca.get(cl, 0) + float(areas[fids == pid].sum())
    ranked = sorted(ca, key=lambda c: -ca[c])[:len(PALETTE)]
    cmap = {cl: PALETTE[i] for i, cl in enumerate(ranked)}
    # colored glb: one submesh per color
    sc = trimesh.Scene()
    colors_used = {}
    for i, cl in enumerate(ranked):
        fmask = np.isin(fids, [p for p, c in eff.items() if c == cl])
        mm = merged.copy()
        mm.update_faces(fmask)
        mm.remove_unreferenced_vertices()
        if not len(mm.faces):
            continue
        name, rgb = cmap[cl]
        mm.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=[rgb[0], rgb[1], rgb[2], 255],
                metallicFactor=0.0, roughnessFactor=0.9))
        sc.add_geometry(mm, geom_name=f"c_{name}")
        colors_used[name] = {"cluster": int(cl),
                             "pids": sorted(int(p) for p, c in eff.items() if c == cl),
                             "area_share": round(ca[cl] / sum(ca.values()), 3)}
    rest = np.isin(fids, [p for p, c in eff.items() if c not in ranked])
    if rest.any():
        mm = merged.copy(); mm.update_faces(rest); mm.remove_unreferenced_vertices()
        mm.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=[128, 128, 128, 255], metallicFactor=0.0, roughnessFactor=0.9))
        sc.add_geometry(mm, geom_name="c_gray")
    sc.export(f"{od}/colored.glb")
    json.dump(colors_used, open(f"{od}/colors.json", "w"), indent=1)

    env = {**os.environ, "OMP_NUM_THREADS": "10",
           "LD_LIBRARY_PATH": "/fsx/sfr/weikaih/trellis2-data-pipeline/env/blender_libs:" + os.environ.get("LD_LIBRARY_PATH", "")}
    tmp = f"{od}/_r"; os.makedirs(tmp, exist_ok=True)
    subprocess.run([BLENDER, "-b", "-t", "12", "-P", SCRIPT, "--", "--object", f"{od}/colored.glb",
                    "--cond_views", json.dumps(VIEWS), "--cond_output_folder", tmp,
                    "--cond_resolution", "512", "--engine", "CYCLES"],
                   capture_output=True, timeout=900, env=env)
    outs = sorted(glob.glob(f"{tmp}/0*.webp"))
    for i, o in enumerate(outs):
        os.replace(o, f"{od}/view_{i}.webp")
    try:
        os.rmdir(tmp)
    except OSError:
        pass

    # panel: photo + 3 colored views + legend bar
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
    def sq(p, s=400):
        im = Image.open(p)
        if im.mode == 'RGBA':
            bg = Image.new('RGBA', im.size, (245, 245, 245, 255)); im = Image.alpha_composite(bg, im)
        return im.convert('RGB').resize((s, s), Image.LANCZOS)
    tiles = [sq(pilot()[sha]['front_view'])] + [sq(f"{od}/view_{i}.webp") for i in range(len(outs))]
    W = 400 * len(tiles); LEG = 56
    panel = Image.new('RGB', (W, 400 + LEG), (250, 250, 250))
    for i, t in enumerate(tiles):
        panel.paste(t, (i * 400, 0))
    dr = ImageDraw.Draw(panel)
    x = 10
    for name, spec in colors_used.items():
        rgb = dict(PALETTE)[name]
        dr.rectangle([x, 412, x + 30, 442], fill=rgb, outline=(0, 0, 0))
        dr.text((x + 36, 414), name, fill=(0, 0, 0), font=font)
        x += 40 + 12 * len(name) + 24
    panel.save(f"{od}/panel.jpg", quality=92)
    print(f"[{sha[:8]}] {len(colors_used)} colored clusters -> {od}/panel.jpg", flush=True)


if __name__ == "__main__":
    for sha in sys.argv[1:]:
        try:
            prep(sha)
        except Exception:
            import traceback; traceback.print_exc()
    print("VLM_MERGE_PREP_DONE")
