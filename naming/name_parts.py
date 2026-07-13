"""Name highlighted parts with Qwen3.6-27B (vLLM, 1 GPU).
Scans out_pairs/*/*/{part_ref,part1_ref,part2_ref,donor_ref}.jpg, asks for a short
part name, writes part_names.json {img_path_rel: name} and fills meta.json
'instruction' from 'instruction_template'.
Usage: python name_parts.py
"""
import os, glob, json, base64, io
from PIL import Image
from vllm import LLM, SamplingParams


def main():

    B = "/fsx/hyperpod/weikaih_edit"
    MODEL = "Qwen/Qwen3.6-27B"

    PROMPT = ("You see a 3D object: the leftmost image is a rendered view of the whole object; "
              "the other three panels are orthographic point views where ONE PART is highlighted in RED "
              "(the rest is gray). Name the RED part in 1-3 lowercase words as you would in an edit "
              "instruction (e.g. 'left front wheel', 'chimney', 'sword'). If unsure, give your best "
              "guess based on position and shape. Output ONLY JSON: {\"part\": \"<name>\"} /no_think")

    imgs = []
    for pat in ['part_ref', 'part1_ref', 'part2_ref', 'donor_ref']:
        imgs += sorted(glob.glob(f"{os.environ.get('PAIRS_ROOT', B + '/out_pairs')}/*/*/{pat}.jpg"))
    print(f"{len(imgs)} part refs to name", flush=True)

    llm = LLM(model=MODEL, max_model_len=8192, gpu_memory_utilization=0.9,
              limit_mm_per_prompt={"image": 1}, dtype="bfloat16")
    sp = SamplingParams(temperature=0.0, max_tokens=128)

    def b64(p):
        im = Image.open(p).convert('RGB')
        im.thumbnail((1400, 1400))
        buf = io.BytesIO(); im.save(buf, format='JPEG', quality=90)
        return base64.b64encode(buf.getvalue()).decode()

    batch = [{
        "prompt": f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{PROMPT}<|im_end|>\n<|im_start|>assistant\n",
        "multi_modal_data": {"image": Image.open(p).convert('RGB')},
    } for p in imgs]
    outs = llm.generate(batch, sp)

    names = {}
    for p, o in zip(imgs, outs):
        t = o.outputs[0].text.strip()
        try:
            t = t[t.index('{'):t.rindex('}') + 1]
            names[os.path.relpath(p, B)] = json.loads(t)['part'].strip().lower()
        except Exception:
            names[os.path.relpath(p, B)] = "part"
    json.dump(names, open(f"{B}/part_names.json", "w"), indent=1)

    # fill instructions in metas
    n = 0
    for mp in glob.glob(f"{os.environ.get('PAIRS_ROOT', B + '/out_pairs')}/*/*/meta.json"):
        d = os.path.dirname(mp)
        meta = json.load(open(mp))
        def nm(base):
            k = os.path.relpath(f"{d}/{base}.jpg", B)
            return names.get(k)
        if 'turns' in meta:  # E10
            p1, p2 = nm('part1_ref') or 'part', nm('part2_ref') or 'part'
            for t, pn in zip(meta['turns'], [p1, p2]):
                t['instruction'] = t['instruction_template'].replace('{PART1}', pn).replace('{PART2}', pn)
        elif 'instruction_template' in meta:
            it = meta['instruction_template']
            it = it.replace('{PART}', nm('part_ref') or 'part')
            it = it.replace('{DONOR_PART}', nm('donor_ref') or nm('part_ref') or 'part')
            meta['instruction'] = it
        json.dump(meta, open(mp, 'w'), indent=1)
        n += 1
    print(f"NAMED {len(names)} refs, filled {n} metas")
    print("NAME_PARTS_DONE")


if __name__ == "__main__":
    main()
