"""E1a step 1: edit front views with Qwen-Image-Edit-2511.
Reads pilot assets + a per-sha instruction json, writes edited_views/<sha>.png (RGBA:
edited RGB + ORIGINAL alpha — E1a freezes geometry so the silhouette is unchanged,
which also lets TRELLIS's alpha-path preprocessing work without rembg).

Usage: python edit_views.py <instructions.json> [shard] [nshards] [limit]
instructions.json: {sha: "edit instruction", ...}
"""
import os, sys, json, time
import numpy as np
import torch
from PIL import Image

INSTR = json.load(open(sys.argv[1]))
SHARD = int(sys.argv[2]) if len(sys.argv) > 2 else 0
NSHARDS = int(sys.argv[3]) if len(sys.argv) > 3 else 1
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9
INSTR = {k: v for i, (k, v) in enumerate(sorted(INSTR.items())) if i % NSHARDS == SHARD}
OUT = os.environ.get("EDIT_OUT", "/fsx/hyperpod/weikaih_edit/edited_views")
os.makedirs(OUT, exist_ok=True)

pilot = {a['sha']: a for a in json.load(open('/fsx/hyperpod/weikaih_edit/pilot200.json'))}

from diffusers import QwenImageEditPlusPipeline
pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511", torch_dtype=torch.bfloat16)
pipe.to("cuda")
print("QIE-2511 loaded", flush=True)

n = 0
for sha, instr in INSTR.items():
    if n >= LIMIT:
        break
    op = f"{OUT}/{sha}.png"
    if os.path.exists(op) or sha not in pilot:
        continue
    t0 = time.time()
    src = Image.open(pilot[sha]['front_view'])
    alpha = src.split()[-1] if src.mode == 'RGBA' else None
    # composite on neutral gray for the editor
    if src.mode == 'RGBA':
        bg = Image.new('RGBA', src.size, (128, 128, 128, 255))
        rgb = Image.alpha_composite(bg, src).convert('RGB')
    else:
        rgb = src.convert('RGB')
    out = pipe(image=[rgb], prompt=instr,
               generator=torch.manual_seed(0),
               true_cfg_scale=4.0, negative_prompt=" ",
               num_inference_steps=40, guidance_scale=1.0,
               num_images_per_prompt=1).images[0]
    out = out.resize(src.size, Image.LANCZOS)
    # reattach original alpha (silhouette frozen for texture edits)
    if alpha is not None:
        out = out.convert('RGB')
        out.putalpha(alpha)
    out.save(op)
    json.dump({"sha": sha, "instruction": instr},
              open(f"{OUT}/{sha}.json", "w"))
    print(f"[{sha}] {time.time()-t0:.1f}s | {instr}", flush=True)
    n += 1
print("EDIT_VIEWS_DONE", flush=True)
