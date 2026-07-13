"""TRUE dual-blind edit verification with Qwen3.6-27B (vLLM, 1 GPU).
Stage A (blind, images only): judge describes what visibly changed — the
instruction is NEVER shown alongside the images.
Stage B (text only): given the blind description and the intended instruction,
rate agreement.
Inputs: manifest json [{key, grid, instruction}]; writes <out>.json with
{key: {observed, visible_change, match, confidence}}.
Usage: python dual_blind_judge.py <manifest.json> <out.json>
"""
import os, sys, json
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

SCHEMA_A = {"type": "object",
            "properties": {"observed": {"type": "string", "maxLength": 400},
                           "visible_change": {"type": "boolean"}},
            "required": ["observed", "visible_change"]}
SCHEMA_B = {"type": "object",
            "properties": {"match": {"enum": ["yes", "partial", "no"]},
                           "confidence": {"enum": ["high", "medium", "low"]}},
            "required": ["match", "confidence"]}

MODEL = "Qwen/Qwen3.6-27B"

PROMPT_A = (
    "This image strip shows renders of ONE 3D object BEFORE an edit (leftmost panel(s)) "
    "and AFTER the edit (rightmost panels). The yellow banner text is a task id only — ignore any words in it.\n"
    "Compare the before and after renders carefully. Describe in one precise sentence what "
    "visibly changed (geometry added / removed / moved / deformed, or material/color change). "
    "If you cannot see any difference between before and after, say exactly 'no visible change'. "
    "Be skeptical: do NOT guess a change you cannot actually see.\n"
    "Output ONLY JSON: {\"observed\": \"<one sentence>\", \"visible_change\": true/false} /no_think")

PROMPT_B = (
    "An editor was asked to apply this edit to a 3D object:\n  INSTRUCTION: \"{instr}\"\n"
    "An independent inspector who did NOT know the instruction examined the before/after renders "
    "and reported:\n  OBSERVED: \"{obs}\" (visible_change={vis})\n"
    "Judge whether the observed change fulfils the instruction. 'yes' = the described change is "
    "clearly the instructed edit; 'partial' = related but incomplete/with side effects; "
    "'no' = unrelated, opposite, or no visible change.\n"
    "Output ONLY JSON: {{\"match\": \"yes|partial|no\", \"confidence\": \"high|medium|low\"}} /no_think")


def parse_json(t):
    try:
        return json.loads(t[t.index('{'):t.rindex('}') + 1])
    except Exception:
        return None


def main():
    manifest = json.load(open(sys.argv[1]))
    outp = sys.argv[2]
    items = [m for m in manifest if os.path.exists(m['grid'])]
    print(f"{len(items)}/{len(manifest)} grids found", flush=True)

    llm = LLM(model=MODEL, max_model_len=8192, gpu_memory_utilization=0.9,
              limit_mm_per_prompt={"image": 1}, dtype="bfloat16")
    sp = SamplingParams(temperature=0.0, max_tokens=512,
                        structured_outputs=StructuredOutputsParams(json=SCHEMA_A))
    sp_b = SamplingParams(temperature=0.0, max_tokens=128,
                          structured_outputs=StructuredOutputsParams(json=SCHEMA_B))

    # Stage A — blind observation
    batch = []
    for m in items:
        im = Image.open(m['grid']).convert('RGB')
        im.thumbnail((2000, 640))
        batch.append({
            "prompt": ("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                       + PROMPT_A + "<|im_end|>\n<|im_start|>assistant\n"),
            "multi_modal_data": {"image": im},
        })
    outs_a = llm.generate(batch, sp)
    obs = []
    for o in outs_a:
        d = parse_json(o.outputs[0].text.strip()) or {}
        obs.append({"observed": d.get('observed', ''),
                    "visible_change": bool(d.get('visible_change', False))})

    # Stage B — text-only match
    batch_b = [{
        "prompt": ("<|im_start|>user\n"
                   + PROMPT_B.format(instr=m['instruction'],
                                     obs=o['observed'][:400],
                                     vis=o['visible_change'])
                   + "<|im_end|>\n<|im_start|>assistant\n")
    } for m, o in zip(items, obs)]
    outs_b = llm.generate(batch_b, sp_b)

    res = {}
    for m, o, ob in zip(items, obs, outs_b):
        d = parse_json(ob.outputs[0].text.strip()) or {"match": "err", "confidence": "low"}
        res[m['key']] = {**o, **d}
    json.dump(res, open(outp, "w"), indent=1)
    ok = sum(1 for v in res.values() if v.get('match') == 'yes')
    nvc = sum(1 for v in res.values() if not v.get('visible_change'))
    print(f"JUDGE_DONE {ok}/{len(res)} match=yes, {nvc} no-visible-change -> {outp}")


if __name__ == "__main__":
    main()
