"""Name highlighted parts in a directory of ref jpgs (generic, vLLM Qwen3.6-27B).
Usage: python name_refs.py <refs_dir> <out.json>   -> {sha: "part name"}
"""
import os, sys, json, glob
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

MODEL = "Qwen/Qwen3.6-27B"
SCHEMA = {"type": "object", "properties": {"part": {"type": "string", "maxLength": 60}},
          "required": ["part"]}
PROMPT = ("You see a 3D object: the leftmost image is a rendered view of the whole object; "
          "the other three panels show only the VISIBLE surface points from front/side/top, "
          "with ONE PART highlighted in RED (the rest is gray). Name the RED part in 1-3 "
          "lowercase words as you would in an edit instruction (e.g. 'left front wheel', "
          "'chimney', 'jaw'). Base the name on the part's position and shape on THIS object. "
          "Output ONLY JSON: {\"part\": \"<name>\"} /no_think")


def main():
    refs_dir, outp = sys.argv[1], sys.argv[2]
    imgs = sorted(glob.glob(f"{refs_dir}/*.jpg"))
    print(f"{len(imgs)} refs", flush=True)
    llm = LLM(model=MODEL, max_model_len=8192, gpu_memory_utilization=0.9,
              limit_mm_per_prompt={"image": 1}, dtype="bfloat16")
    sp = SamplingParams(temperature=0.0, max_tokens=96,
                        structured_outputs=StructuredOutputsParams(json=SCHEMA))
    batch = [{"prompt": ("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                         + PROMPT + "<|im_end|>\n<|im_start|>assistant\n"),
              "multi_modal_data": {"image": Image.open(p).convert('RGB')}} for p in imgs]
    outs = llm.generate(batch, sp)
    res = {}
    for p, o in zip(imgs, outs):
        t = o.outputs[0].text.strip()
        try:
            res[os.path.basename(p)[:-4]] = json.loads(t)['part'].strip().lower()
        except Exception:
            res[os.path.basename(p)[:-4]] = "part"
    json.dump(res, open(outp, "w"), indent=1)
    print(f"NAME_REFS_DONE {len(res)}")


if __name__ == "__main__":
    main()
