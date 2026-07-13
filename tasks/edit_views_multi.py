"""Run QIE-2511 once, process multiple (instructions.json -> out_dir) jobs.
Usage: python edit_views_multi.py <shard> <nshards> <instr1:out1> [<instr2:out2> ...]
"""
import os, sys, json, time
import torch
from PIL import Image

SHARD, NSHARDS = int(sys.argv[1]), int(sys.argv[2])
JOBS = [a.split(':') for a in sys.argv[3:]]
B = "/fsx/hyperpod/weikaih_edit"
pilot = {a['sha']: a for a in json.load(open(f'{B}/pilot200.json'))}

from diffusers import QwenImageEditPlusPipeline
pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511", torch_dtype=torch.bfloat16)
pipe.to("cuda")
print("QIE-2511 loaded", flush=True)

for instr_file, out_dir in JOBS:
    os.makedirs(out_dir, exist_ok=True)
    instrs = sorted(json.load(open(instr_file)).items())[SHARD::NSHARDS]
    for sha, instr in instrs:
        op = f"{out_dir}/{sha}.png"
        if os.path.exists(op) or sha not in pilot:
            continue
        t0 = time.time()
        src = Image.open(pilot[sha]['front_view'])
        alpha = src.split()[-1] if src.mode == 'RGBA' else None
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
        if alpha is not None:
            out = out.convert('RGB')
            out.putalpha(alpha)
        out.save(op)
        json.dump({"sha": sha, "instruction": instr}, open(f"{out_dir}/{sha}.json", "w"))
        print(f"[{os.path.basename(out_dir)}/{sha[:8]}] {time.time()-t0:.1f}s", flush=True)
print("EDIT_VIEWS_MULTI_DONE", flush=True)
