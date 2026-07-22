# 3D Edit Engine — instruction-editing pair factory for 3D assets

Generates **(before-asset, instruction, after-asset)** training pairs across a 12-type edit
taxonomy, to train an instruction-driven 3D editing model. This supervision does not exist
on the web — it must be constructed. The engine uses only **frozen off-the-shelf models** —
no training anywhere inside it.

| Role | Model / tool |
|---|---|
| 3D generation (texture-given-shape, full image-to-3D) | **TRELLIS.2-4B** (microsoft, frozen) |
| 2D instruction-driven image editing | **Qwen-Image-Edit-2511** |
| Native 3D part segmentation | **P3-SAM** (tencent/Hunyuan3D-Part) |
| In-context part re-synthesis (E4 only) | **X-Part** (tencent/Hunyuan3D-Part) |
| Part naming / semantic merging / dual-blind verify / captions | **Qwen3.6-27B** (vLLM) |
| Exact geometric edits, texture rebake, PBR channel edits | plain `trimesh`/`numpy` (no model) |

Live QC viewer (every pair, verdict, 2D strips, drag-to-rotate GLBs):
`https://dashboard.ai-research-wk.com/reports/3d-generation-editing/editing-pairs-viewer`

---

## Design logic (why it's built this way)

**Grand principle:** *what code can construct exactly, never generate; what must be generated,
only touch the part that should change; every pair must be auto-verifiable.* The goal is
supervision that is **constructed, not generated** wherever possible.

### 1. Split edits by nature → two tracks
- **Appearance edits (material / color / style) → generative.** Edit a rendered view with QIE,
  lift it back to 3D with TRELLIS. Trick: **freeze geometry, re-sample only the texture flow**
  → after ≡ before geometrically (silhouette IoU 0.99), the only change is what the instruction said.
- **Structural edits (remove / add / move / duplicate / deform / replace) → procedural.** P3-SAM
  gives per-face part ids; the rest is deterministic geometry on the real GLB. **The `after` is
  mathematically correct** — zero noise, zero hallucination, the cleanest supervision.

### 2. Every new problem: quantify first, then use the lightest method
Not feature-piling — measure severity, then match investment:
- **Text/logo mangled by generation** → measured **only ~8% of assets** have significant text →
  don't complicate the pipeline; **demote to filtering** (dual-blind judge drops the bad ones).
- **Part removal exposes inner walls ("looks like a hole")** → measured **~17% severe** →
  **demote to detection + gating** (`interior_exposure` metric in the geometry gate).
- **"Add part" unrealistic (only re-adds random original parts)** → use **semantic
  reverse-of-removal** (remove a real hat/wings/tail → reverse) → **93% blind-judge, no X-Part**.

None of these got over-engineered (no text-inpainting, no hole-capping, no new training) —
quantification showed they're edge cases where filtering/detection suffices.

### 3. Recurring "right tools" (sedimented principles)
1. **Reverse pairs** — remove↔add, scale↔unscale, edit↔restore. Free bidirectional data,
   geometry exact by construction, prevents mode collapse ("only adds, never removes").
2. **Two-layer part stack** (forced by the "eyes left behind when moving the head" bug):
   - **geometry layer** handles what the VLM can't SEE — interior parts (winding-number
     containment), double walls (nested-shell clustering), split-surface seams (tangent continuity);
   - **VLM layer** handles what geometry can't KNOW — rider-vs-dragon semantics (color-code the
     clusters, one Qwen call groups + names them). Each covers the other's blind spot.
3. **The unedited region never goes through a generative model.** Use **`rebake_texture.py`**
   (deterministic per-texel transfer from the original) — because generative models optimize a
   *generation* objective, not a *reconstruction* one, so they can't guarantee pixel-exact preservation.
4. **The right tool, not the bigger tool.** Verified the hard way: Hunyuan-Paint is not a text
   silver bullet (all generative texturing loses text); X-Part is wrong for "add a part" (it only
   grows geometry the object already implies, not a semantic "hat"); tileable material libraries
   look fake at scale. The best result usually comes from **decomposing the problem correctly**
   ("metal = change metallic/roughness, not color"; "add a hat = inverse of removing a hat"),
   not from a heavier model.

### 4. Auto-verification is the precondition for scale
125 pairs can be eyeballed; 470K cannot. Every pair passes **two gates**:
- **geometry hard checks** (CPU): frozen-geometry silhouette IoU, part visibility, orphan
  detection, clean-cut interior-exposure.
- **dual-blind judge**: stage-A describes the change from images ONLY (instruction hidden),
  stage-B matches text-only against the instruction. Blindness is essential — shown the
  instruction, the judge confirms mangled pairs. Thresholds calibrated on the 125 human verdicts.

---

## Task matrix & pilot results (dual-blind judged)

| Task | Method | Pilot pass |
|---|---|---|
| E1a global material | freeze geometry + re-sample TRELLIS.2 texture flow on QIE-edited view | 29/30 |
| E1a_rev restore | free inverse of E1a; after = original glb (text perfect) | 25/30 |
| E1b local material | VLM semantic-unit mask + targeted QIE + **rebake outside** | 8-9/11 |
| E1c pattern-keep recolor | *deferred to phase-2 FlowEdit* (fallback ceiling 2/10) | — |
| E2 add part | **semantic reverse-of-removal** ("add a hat/wings/tail") | **28/30** |
| E3 remove part | procedural on VLM semantic unit + **clean-cut gate** | 12/12 |
| E4 replace part | procedural swap → X-Part re-synth → TRELLIS.2 re-texture | 4/6 |
| E5 stylize | QIE-edited view → full TRELLIS.2 image-to-3D regen | 8/8 |
| E6 move/scale part | procedural affine on semantic unit | 19-20/24 |
| E7 duplicate/mirror | procedural copy/mirror of semantic unit | 7/10 |
| E8 cross-asset part | procedural donor paste keeping donor texture | 3/8 |
| E9 deform | twist/taper/bulge/bend on clean assets | 4/8 |
| E10 multi-turn | chains composed from gate-passed single edits | 5+2p/8 |

Notable findings this round:
- **PBR channel-split**: "make it metal/gold/glass" = change `metallicFactor`/`roughnessFactor`,
  keep `baseColorTexture` (which holds the text) untouched → metal look + crisp text, CPU-instant,
  no generation. Covers uniform surface-property edits; color-change edits still need generation.
- **Text is 8% edge case**, routed/filtered. **Interior exposure is ~17%**, gated. **Add-part
  needs no X-Part** — semantic reverse gives 93%.

## Pipeline anatomy

```
selection/   pick assets (VLM-judge axes) + parts (area-share + z-buffer visibility)
tasks/       per-task pair generators: procedural exact ops / QIE+TRELLIS generative /
             X-Part chain (E4) / reverse pairs / semantic add (E2) / semantic-group procedural (v3)
engine/      shared libs: part selection + co-move groups (edit_parts_lib, build_part_tree),
             UV-space local texture merge (e1b_merge), deterministic texture rebake (rebake_texture)
naming/      color-code clusters → Qwen semantic merge+naming (vlm_merge_*), part naming
verify/      auto-gate: geometry checks (IoU, visibility, orphan, clean-cut) + true dual-blind judge
render/      blender CPU renders (bbox-anchored framing, look-at-part closeups), QC grids
viewer/      builds the public model-viewer QC site
slurm/       single-node 8xH200 job scripts + node-hold pattern (hold_nodes + run_on_hold)
configs/     instruction sets + example artifacts (part names, human verdicts)
docs/        RESULTS / GOTCHAS / THIRD_PARTY_PATCHES / REFERENCES
```

### Key verified invariants
1. P3-SAM `face_ids` map 1:1 onto trimesh scene-dump merged face order (bitwise) — masks apply directly.
2. Frozen geometry ⇒ TRELLIS.2 remesh+UV deterministic ⇒ local material edit = pure CPU UV merge; IoU≈0.99.
3. TRELLIS.2 outputs live in normalized [-0.5,0.5] — normalize the original into that frame before any transfer.
4. Edit locality: whole-object re-texture is right when the edit should look native (E4), wrong when it should stay distinct (E8).
5. Removing a part rarely creates a true hole (assets are separate interpenetrating shells) but often exposes interior — detect via inward-facing-normal fraction near the cut.

## Docs
- `docs/RESULTS.md` — per-round quality progression and what fixed what
- `docs/GOTCHAS.md` — every trap (coordinate frames, HF offline, vLLM spawn, text fidelity, interior exposure, …)
- `docs/THIRD_PARTY_PATCHES.md` — required edits to TRELLIS.2 / Hunyuan3D-Part checkouts
- `docs/REFERENCES.md` — frozen models + the editing papers that shaped the taxonomy
