# Pilot results — round-by-round quality progression

All pairs and verdicts are browsable on the QC viewer:
`https://dashboard.ai-research-wk.com/reports/3d-generation-editing/editing-pairs-viewer`

## Round 1 (2026-07-10) — first instantiation of all 12 tasks, 125 pairs
Claude eyeball-verified every pair: **64 pass / 47 marginal / 14 fail.**
- Strong out of the gate: E1a 29/30 (after the good-view fix), E5 8/8, E9 deforms.
- Weak: E3 3/10 (part chosen by FACE-count share broke on low-poly assets; invisible parts),
  E1b 2/11 (mask misplacement + generic part names), E2 donor-paste 1/8, E10 name collisions.

## Round 2 (2026-07-11) — fixed part selection + reverse pairs + auto-gate, 92 pairs
- Part selection v2: surface-AREA share + 6-view z-buffer visibility + part_complexity gate
  → **E3 3/10 → 9/10** (blind judge).
- Reverse pairs for free: E2 = E3⁻¹ (10), E6⁻¹ (10), E1a⁻¹ with caption-driven
  "restore original appearance" instructions (30).
- Auto-gate built and calibrated against the 125 human verdicts:
  geometry checks (E1a silhouette IoU mean **0.991** — frozen geometry proven) +
  TRUE dual-blind judge (single-prompt judging was instruction-biased: it accepted 8/14 known
  fails; the blind two-stage version rejects them).

## Round 2.5 (2026-07-11) — E1b targeted edits
- Targeted QIE ("change only the <named part> …") + visible-part selection + the
  **coordinate-frame mask fix** (normalize original into TRELLIS's [-0.5,0.5] frame before
  nearest-face transfer) → **E1b 2/11 → 9/11**.
- X-Part validated: `pipeline(mesh, aabb=custom)` re-synthesizes a bbox region in context (3/3).

## Round 3 (2026-07-12) — weak-task repair, final pilot state
- E7 5/8 → **7/8** via look-at-part closeup panels in the judge grids.
- E4 full chain (procedural swap → X-Part re-synth → TRELLIS.2 re-texture): **4/6**.
- **E8 X-Part chain REJECTED (1/8)**: whole-object re-texture destroys edit locality; cross-asset
  attach *needs* the donor's alien texture. Reverted to procedural paste.
  → Design principle: re-texture when the edit should look native (replace), never when it
  should stay distinct (cross-asset).
- E10 v2 composed only from gate-passed pairs: 2 yes + 3 partial, zero fails.
- E9 v2.1 (clean assets, stronger ops) still 4/8: twist/bend read as rotation/pose-change from
  fixed views → per-op asset routing is the known fix (mini-production TODO).
- E1c officially deferred to phase-2 FlowEdit (fallback ceiling 2/10), excluded from metrics.

## Final blind-judge scoreboard (116 gated pairs)

| E1a | E1a_rev | E1b | E2 | E3 | E4 | E5 | E6 | E7 | E8* | E9 | E10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 29/30 | 25/30 | 9/11 | 6/10 | 9/10 | 4/6 | 8/8 | 17/20 | 7/8 | 3/8+5M* | 4/8 | 2+3p/5 |

\* E8 = human verdicts on the procedural version (chain reverted).

## Open items for mini-production
1. Per-asset part-name dictionary (name all usable parts once; reuse; enforce distinct names).
2. Category-consistency check on part names (reject "backrest" on an animal skull).
3. Per-op asset routing for E9 (twist ⇒ wide objects, bend ⇒ tall objects).
4. E8 seam improvement without re-texture: X-Part geometry + E1b-style bbox-local texture merge.
5. QIE-2511-Lightning (8-step) for ~5-8x cheaper 2D edits; TRELLIS GPU renders for QC at scale.
6. Look-at-part cameras as the default judge view; instruction paraphrase + material vocabulary
   sampled from texture captions.
