"""Build the interactive 12-task editing-pairs QC site for the viz hub.
- collects pairs from out_pairs/* and out_pilot/{edit_v2,e1c,e5} (fallback edit=v1)
- draco-compresses before/after glbs into the hub assets dir (size-capped)
- copies QC grid jpgs
- embeds my per-pair verdicts from verdicts.json (pending if absent)
- writes reports/3d-generation-editing/editing-pairs-viewer.html
Usage: python build_site.py
Re-run any time; idempotent (skips existing compressed glbs).
"""
import os, json, glob, subprocess, shutil, html

B = "/fsx/hyperpod/weikaih_edit"
HUB = "/fsx/sfr/weikaih/research-proj-report"
RPT = f"{HUB}/reports/3d-generation-editing"
AST = f"{RPT}/assets-editing-pairs"
os.makedirs(AST, exist_ok=True)
BUNX = "/fsx/home/weikai.huang/.bun/bin/bunx"
MAX_GLB_MB = 24          # Pages hard limit is 25MB/file
PAIR_3D_BUDGET_MB = 45   # skip 3D for pairs whose compressed sum exceeds this
PER_TASK_3D = 8

verd = {}
if os.path.exists(f"{B}/verdicts.json"):
    verd = json.load(open(f"{B}/verdicts.json"))

TASK_SHORT = {
    'E1a': 'Material', 'E1b': 'Local material', 'E1c': 'Pattern recolor (deferred)',
    'E2': 'Add part', 'E3': 'Remove part', 'E4': 'Replace part',
    'E5': 'Stylize', 'E6': 'Move / scale part', 'E7': 'Duplicate part',
    'E8': 'Cross-asset part', 'E9': 'Deform', 'E10': 'Multi-turn', 'E1a_rev': 'Restore (reverse)',
}
TASK_INFO = {
    'E1a': ('Global material change', 'freeze geometry + re-sample TRELLIS.2 texture flow on the QIE-edited view'),
    'E1b': ('Part-local material change', 'v2: TARGETED QIE edit ("change only the <part>") + UV-space texture merge by P3-SAM mask — blind-judge 9/11'),
    'E1c': ('Pattern-preserving appearance — DEFERRED', 'fallback route hit its ceiling (2/10); real solution is FlowEdit on the texture flow, scheduled phase-2; excluded from pilot gate metrics'),
    'E2':  ('Part addition', 'r3: exact inverse of SEMANTIC-unit E3 removal (VLM-named groups) — blind-judge 11/12'),
    'E3':  ('Part removal', 'r3: removes a VLM-named semantic unit (co-move group) — blind-judge 12/12'),
    'E4':  ('Part replacement', 'full chain: procedural swap → X-Part in-context re-synthesis → TRELLIS.2 re-texture'),
    'E5':  ('Global stylization', 'QIE-edited view → full TRELLIS.2 image-to-3D regeneration'),
    'E6':  ('Part rigid transform', 'r3: affine on semantic units, instructions from VLM group names'),
    'E7':  ('Duplicate / mirror part', 'r3: copy/mirror whole semantic units'),
    'E8':  ('Cross-asset composition', 'procedural donor paste keeping donor texture (X-Part chain rejected: re-texture broke edit locality)'),
    'E9':  ('Mesh deformation', 'v2.1: clean assets only, stronger twist/taper/bulge + bend'),
    'E10': ('Multi-turn chains', 'r3: two-step chains over distinct semantic units'),
    'E1a_rev': ('Reverse material pairs', 'free inverse of E1a: edited asset + caption-driven restore-original instruction'),
}

def draco(src, dst):
    if os.path.exists(dst):
        return os.path.getsize(dst) / 1e6
    r = subprocess.run([BUNX, "--yes", "@gltf-transform/cli", "draco", src, dst],
                       capture_output=True, timeout=600)
    if r.returncode != 0 or not os.path.exists(dst):
        return None
    mb = os.path.getsize(dst) / 1e6
    if mb > MAX_GLB_MB:
        os.remove(dst)
        return None
    return mb

pilot = {a['sha']: a for a in json.load(open(f'{B}/pilot200.json'))}
pairs = []

def add_pair(task, name, instr, glbs, grid, extra_img=None):
    pairs.append({"task": task, "name": name, "instr": instr,
                  "glbs": glbs, "grid": grid, "extra": extra_img})

# --- out_pairs (r1) + out_pairs2 (r2) + out_pairs3 (r3: SEMANTIC edit units)
GEO_SRC = {"r2": f"{B}/out_pairs2/auto_geo.json", "r3": f"{B}/out_pairs3/auto_geo.json"}
JUDGE_SRC = {"r2": f"{B}/judge_round2_results.json", "r3": f"{B}/judge_v3_results.json"}
auto_geo_by, auto_judge_by = {}, {}
for rnd in ("r2", "r3"):
    auto_geo_by[rnd] = json.load(open(GEO_SRC[rnd])) if os.path.exists(GEO_SRC[rnd]) else {}
    auto_judge_by[rnd] = json.load(open(JUDGE_SRC[rnd])) if os.path.exists(JUDGE_SRC[rnd]) else {}

# earlier rounds superseded by fixed later versions are NOT shown
SUPERSEDED_R1 = {'E1b', 'E2', 'E3', 'E4', 'E6', 'E7', 'E9', 'E10'}
SUPERSEDED_R2 = {'E2', 'E3', 'E6', 'E7', 'E10'}      # replaced by r3 semantic units
for root, rnd in [(f"{B}/out_pairs", "r1"), (f"{B}/out_pairs2", "r2"), (f"{B}/out_pairs3", "r3")]:
    for mp in sorted(glob.glob(f"{root}/*/*/meta.json")):
        d = os.path.dirname(mp)
        task = d.split('/')[-2]
        name = os.path.basename(d)
        if task == 'compare':
            continue
        if rnd == 'r1' and task in SUPERSEDED_R1:
            continue
        if rnd == 'r2' and task in SUPERSEDED_R2:
            continue
        meta = json.load(open(mp))
        if task == 'E10':
            instr = "  ▸  ".join((t.get('instruction') or t.get('instruction_template', '?')) for t in meta['turns'])
            glbs = [f"{d}/state{i}.glb" for i in range(3)]
        elif task == 'E1b':
            instr = meta.get('instruction') or meta.get('instruction_template', '?')
            glbs = [pilot[meta['sha']]['glb'], f"{d}/after.glb"]
        else:
            instr = meta.get('instruction', meta.get('instruction_template', '?'))
            glbs = [f"{d}/before.glb", f"{d}/after.glb"]
        grid = f"{root}/compare/{task}__{name}.jpg"
        p_name = name if rnd == 'r1' else f"{rnd}_{name}"
        add_pair(task, p_name, instr, glbs, grid if os.path.exists(grid) else None)
        if rnd in ('r2', 'r3'):
            key = f"{task}/{name}"
            g = auto_geo_by[rnd].get(key, {})
            j = auto_judge_by[rnd].get(key, {})
            v = 'pending'
            note_parts = []
            if g:
                note_parts.append("geo:" + ("ok" if g.get('geo_pass') else "FAIL"))
            if j:
                m2v = {"yes": "pass", "partial": "marginal", "no": "fail"}
                v = m2v.get(j.get('match'), 'pending')
                note_parts.append(f"blind-judge observed: {j.get('observed','')[:140]}")
            if g and not g.get('geo_pass'):
                v = 'fail'
            pairs[-1]['auto'] = {"verdict": v, "note": "auto-gate | " + " | ".join(note_parts)}

# --- pilot-output tasks (E1a / E1c / E5)
def pilot_task(task, edit_dir, views_dir, cmp_dir):
    for md in sorted(glob.glob(f"{edit_dir}/*/meta.json")):
        m = json.load(open(md))
        if m.get('error'):
            continue
        sha = m['sha']
        d = os.path.dirname(md)
        ij = f"{views_dir}/{sha}.json"
        instr = json.load(open(ij))['instruction'].split('. Keep')[0] if os.path.exists(ij) else '?'
        grid = f"{cmp_dir}/{sha}.jpg"
        add_pair(task, sha, instr, [pilot[sha]['glb'], f"{d}/after.glb"],
                 grid if os.path.exists(grid) else None,
                 extra_img=f"{views_dir}/{sha}.png")

e1a_dir = f"{B}/out_pilot/edit_v2" if glob.glob(f"{B}/out_pilot/edit_v2/*/meta.json") else f"{B}/out_pilot/edit"
e1a_views = f"{B}/edited_views_v2" if 'v2' in e1a_dir else f"{B}/edited_views"
e1a_cmp = f"{B}/out_pilot/compare_e1a_v2" if 'v2' in e1a_dir else f"{B}/out_pilot/compare_edit"
pilot_task('E1a', e1a_dir, e1a_views, e1a_cmp)
pilot_task('E1c', f"{B}/out_pilot/e1c", f"{B}/e1c_views", f"{B}/out_pilot/compare_e1c")
pilot_task('E5', f"{B}/out_pilot/e5", f"{B}/e5_views", f"{B}/out_pilot/compare_e5")


# --- compress glbs + copy grids
print(f"{len(pairs)} pairs collected")
n3d = {}
for p in pairs:
    t = p['task']
    p['id'] = f"{t}__{p['name']}"
    key = f"{t}/{p['name']}"
    if p.get('auto'):
        p['verdict'] = p['auto']['verdict']
        p['note'] = p['auto']['note']
    else:
        p['verdict'] = verd.get(key, {}).get('verdict', 'pending')
        p['note'] = verd.get(key, {}).get('note', '')
    # grid copy
    if p['grid']:
        gdst = f"{AST}/g_{p['id']}.jpg"
        if not os.path.exists(gdst):
            shutil.copy(p['grid'], gdst)
        p['grid_rel'] = f"assets-editing-pairs/g_{p['id']}.jpg"
    else:
        p['grid_rel'] = None
    if p.get('extra') and os.path.exists(p['extra']):
        edst = f"{AST}/x_{p['id']}.png"
        if not os.path.exists(edst):
            shutil.copy(p['extra'], edst)
        p['extra_rel'] = f"assets-editing-pairs/x_{p['id']}.png"
    else:
        p['extra_rel'] = None
    # 3D (cap per task)
    p['glb_rels'] = None
    if n3d.get(t, 0) >= PER_TASK_3D:
        continue
    rels, tot, ok = [], 0, True
    for i, g in enumerate(p['glbs']):
        if not os.path.exists(g):
            ok = False; break
        dst = f"{AST}/{p['id']}__{i}.glb"
        mb = draco(g, dst)
        if mb is None:
            ok = False; break
        tot += mb
        rels.append(f"assets-editing-pairs/{p['id']}__{i}.glb")
    if ok and tot <= PAIR_3D_BUDGET_MB:
        p['glb_rels'] = rels
        n3d[t] = n3d.get(t, 0) + 1
        print(f"  3D {p['id']} {tot:.1f}MB")
    else:
        for i in range(len(p['glbs'])):
            f = f"{AST}/{p['id']}__{i}.glb"
            if os.path.exists(f):
                os.remove(f)

# --- html
cards = {}
order = ['E1a', 'E1a_rev', 'E1b', 'E1c', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10']
for t in order:
    cards[t] = [p for p in pairs if p['task'] == t]

pdata = [{k: p.get(k) for k in ['id', 'task', 'instr', 'verdict', 'note', 'grid_rel', 'extra_rel', 'glb_rels']}
         for p in pairs]

page = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Editing Data Engine — 12-task pilot pairs (interactive QC)</title>
<script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"></script>
<style>
body{font-family:-apple-system,'Segoe UI',Roboto,sans-serif;background:#0f1115;color:#e6e6e6;margin:0;padding:0 0 80px}
header{padding:28px 34px 10px}h1{font-size:22px;margin:0 0 6px}
.sub{color:#9aa4b2;font-size:13.5px;max-width:980px;line-height:1.5}
.tabs{display:flex;flex-wrap:wrap;gap:8px;padding:16px 34px}
.tab{padding:7px 14px;border-radius:18px;background:#1b2230;cursor:pointer;font-size:13px;border:1px solid #2a3446}
.tab.active{background:#2f6feb;border-color:#2f6feb;color:#fff}
.tab .n{opacity:.7;font-size:11px;margin-left:4px}
.tinfo{padding:0 34px 8px;color:#9aa4b2;font-size:13px}
.cards{display:flex;flex-direction:column;gap:18px;padding:10px 34px}
.card{background:#161b24;border:1px solid #232c3d;border-radius:12px;overflow:hidden}
.chead{display:flex;align-items:center;gap:10px;padding:12px 16px;font-size:13.5px}
.badge{font-size:11px;font-weight:700;padding:3px 9px;border-radius:10px;text-transform:uppercase}
.badge.pass{background:#1d4028;color:#7ee2a0}.badge.marginal{background:#453b14;color:#ffd75e}
.badge.fail{background:#4a1d1d;color:#ff9a9a}.badge.pending{background:#2a3446;color:#9aa4b2}
.tchip{font-size:11px;font-weight:700;padding:3px 9px;border-radius:10px;background:#26304a;color:#9fc0ff;white-space:nowrap}
.r2chip{font-size:10px;padding:2px 8px;border-radius:9px;background:#203a2c;color:#8fd6a8;white-space:nowrap}
.instr{flex:1}.note{color:#9aa4b2;font-size:12px;padding:0 16px 10px}
.gridimg{width:100%;display:block;background:#000}
.b3d{margin:10px 16px 14px;padding:7px 16px;border-radius:8px;background:#22304a;color:#cfe1ff;border:1px solid #31456b;cursor:pointer;font-size:13px}
.viewers{display:flex;gap:10px;padding:0 16px 16px;flex-wrap:wrap}
.vwrap{flex:1;min-width:300px}.vlabel{font-size:12px;color:#9aa4b2;padding:4px 2px}
model-viewer{width:100%;height:380px;background:#0a0c10;border-radius:8px}
.extra{max-height:220px;border-radius:8px;margin:0 16px 10px}
</style></head><body>
<header><h1>Editing Data Engine — 12-task pilot pairs</h1>
<div class="sub">Every task of the 12-type edit taxonomy instantiated at pilot scale with the frozen
TRELLIS.2-4B engine + P3-SAM parts + Qwen-Image-Edit-2511 + Qwen3.6-27B (part naming &amp; dual-blind verify).
Round 1 = claude-verified (125 pairs: 64 pass / 47 marginal / 14 fail). Round 2 ("round 2" chip) = fixed part
selection + reverse pairs + targeted local-material edits, gated automatically (geometry 137/138; true dual-blind
judge — Remove part 9/10, Local material 9/11, combined accept 76%). Each card: instruction, verdict, 2D QC strip,
drag-to-rotate before/after GLBs.</div></header>
<div class="tabs" id="tabs"></div><div class="tinfo" id="tinfo"></div><div class="cards" id="cards"></div>
<script>
const PAIRS=__PAIRS__;
const TASKS=__TASKS__;
const INFO=__INFO__;
const SHORT=__SHORT__;
let cur=Object.keys(TASKS)[0];
function esc(s){const d=document.createElement('div');d.textContent=s||'';return d.innerHTML}
function render(){
  const tabs=document.getElementById('tabs');tabs.innerHTML='';
  for(const t of Object.keys(TASKS)){
    const el=document.createElement('div');el.className='tab'+(t===cur?' active':'');
    const ps=PAIRS.filter(p=>p.task===t);
    const np=ps.filter(p=>p.verdict==='pass').length;
    el.innerHTML=esc(SHORT[t]||t)+' <span class="n">'+esc(t)+' · '+ps.length+(np?' · '+np+'✓':'')+'</span>';
    el.onclick=()=>{cur=t;render()};tabs.appendChild(el);
  }
  document.getElementById('tinfo').innerHTML='<b>'+esc(SHORT[cur]||cur)+' ('+esc(cur)+') — '+esc(INFO[cur][0])+'</b> · '+esc(INFO[cur][1]);
  const c=document.getElementById('cards');c.innerHTML='';
  for(const p of PAIRS.filter(p=>p.task===cur)){
    const card=document.createElement('div');card.className='card';
    let h='<div class="chead"><span class="tchip">'+esc(SHORT[p.task]||p.task)+'</span><span class="badge '+p.verdict+'">'+p.verdict+'</span><span class="instr">'+esc(p.instr)+'</span>'+(p.id.startsWith(p.task+'__r2_')?'<span class="r2chip">round 2</span>':'')+'</div>';
    if(p.note)h+='<div class="note">'+esc(p.note)+'</div>';
    if(p.grid_rel)h+='<a href="'+p.grid_rel+'" target="_blank"><img class="gridimg" loading="lazy" src="'+p.grid_rel+'"></a>';
    if(p.glb_rels)h+='<button class="b3d" onclick="load3d(this,\\''+p.id+'\\')">Load 3D (drag to rotate)</button><div class="viewers" id="v_'+p.id+'"></div>';
    card.innerHTML=h;c.appendChild(card);
  }
  window.scrollTo(0,0);
}
function load3d(btn,id){
  const p=PAIRS.find(x=>x.id===id);const v=document.getElementById('v_'+id);
  const labels=p.glb_rels.length===3?['state 0 (before)','state 1 (after turn 1)','state 2 (after turn 2)']:['before','after'];
  v.innerHTML=p.glb_rels.map((g,i)=>'<div class="vwrap"><div class="vlabel">'+labels[i]+'</div><model-viewer src="'+g+'" camera-controls auto-rotate shadow-intensity="0.8" exposure="0.9"></model-viewer></div>').join('');
  btn.remove();
}
render();
</script></body></html>"""

page = page.replace('__PAIRS__', json.dumps(pdata))
page = page.replace('__TASKS__', json.dumps({t: len(cards[t]) for t in order}))
page = page.replace('__INFO__', json.dumps(TASK_INFO))
page = page.replace('__SHORT__', json.dumps(TASK_SHORT))
open(f"{RPT}/editing-pairs-viewer.html", "w").write(page)

import subprocess as sp
sz = sp.run(['du', '-sh', AST], capture_output=True, text=True).stdout.split()[0]
print(f"HTML written; assets dir {sz}; 3D per task: {n3d}")
print("BUILD_SITE_DONE")
