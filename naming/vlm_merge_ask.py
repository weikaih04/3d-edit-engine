"""VLM semantic-merge prototype, step 2 (GPU, vLLM Qwen3.6-27B):
Given the color-coded cluster panel, ask the VLM to group colors into nameable
semantic parts. Guided JSON. Writes vlm_merge/<sha>/groups.json
Usage: python vlm_merge_ask.py <sha> [sha ...]
"""
import os, sys, json
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

B = "/fsx/hyperpod/weikaih_edit"
MODEL = "Qwen/Qwen3.6-27B"

SCHEMA = {"type": "object",
          "properties": {"object": {"type": "string", "maxLength": 60},
                         "groups": {"type": "array", "maxItems": 12, "items": {
                             "type": "object",
                             "properties": {"name": {"type": "string", "maxLength": 40},
                                            "colors": {"type": "array", "maxItems": 12,
                                                       "items": {"type": "string", "maxLength": 12}}},
                             "required": ["name", "colors"]}}},
          "required": ["object", "groups"]}

PROMPT_TMPL = (
    "The leftmost panel is a photo of a 3D object. The other panels show the SAME object with its "
    "geometry clusters painted in flat colors (legend at the bottom; gray = small leftover pieces).\n"
    "Available colors: {colors}.\n"
    "Task: group the colors into SEMANTIC PARTS a person would name when editing this object "
    "(e.g. head, body, hat, weapon, base). Rules:\n"
    "- every listed color appears in exactly one group;\n"
    "- colors that belong to the same nameable thing go together (e.g. if the head is painted in "
    "two colors, both go in the 'head' group);\n"
    "- name each group in 1-3 lowercase words specific to THIS object;\n"
    "- also name the whole object.\n"
    "Output ONLY JSON. /no_think")


def main():
    shas = sys.argv[1:]
    llm = LLM(model=MODEL, max_model_len=8192, gpu_memory_utilization=0.9,
              limit_mm_per_prompt={"image": 1}, dtype="bfloat16")
    sp = SamplingParams(temperature=0.0, max_tokens=512,
                        structured_outputs=StructuredOutputsParams(json=SCHEMA))
    batch, metas = [], []
    for sha in shas:
        od = f"{B}/vlm_merge/{sha}"
        colors = json.load(open(f"{od}/colors.json"))
        im = Image.open(f"{od}/panel.jpg").convert('RGB')
        im.thumbnail((2000, 640))
        batch.append({"prompt": ("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                                 + PROMPT_TMPL.format(colors=", ".join(colors))
                                 + "<|im_end|>\n<|im_start|>assistant\n"),
                      "multi_modal_data": {"image": im}})
        metas.append((sha, od, colors))
    outs = llm.generate(batch, sp)
    for (sha, od, colors), o in zip(metas, outs):
        t = o.outputs[0].text.strip()
        try:
            d = json.loads(t[t.index('{'):t.rindex('}') + 1])
        except Exception:
            d = {"object": "?", "groups": [], "raw": t[:300]}
        json.dump(d, open(f"{od}/groups.json", "w"), indent=1)
        gs = "; ".join(f"{g['name']}={'+'.join(g['colors'])}" for g in d.get('groups', []))
        print(f"[{sha[:8]}] {d.get('object')}: {gs}", flush=True)
    print("VLM_MERGE_ASK_DONE")


if __name__ == "__main__":
    main()
