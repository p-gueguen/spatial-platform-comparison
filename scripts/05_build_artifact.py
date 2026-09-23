#!/usr/bin/env python3
"""Build the self-contained comparison Artifact (HTML) from recomputed metrics + pricing model."""
import os, json, base64, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT="/srv/GT/analysis/pgueguen/spatial_platform_comparison"
OUT=f"{ROOT}/outputs"; FIG=f"{OUT}/figs"
SCRATCH=f"{OUT}"   # report is written next to the metrics it is built from
os.makedirs(SCRATCH,exist_ok=True)
D=json.load(open(f"{OUT}/bundle.json"))["datasets"]
# Shared-set sizes are hardcoded in the prose below; guard against silent drift on re-run.
L1N=len(open(f"{OUT}/shared_genes_L1.txt").read().split())   # flagship whole-tx core
L2N=len(open(f"{OUT}/shared_genes_L2.txt").read().split())   # high-plex L2 (incl. Prime 5K)
assert (L1N,L2N)==(304,4843), f"shared-set sizes changed to L1={L1N} L2={L2N}; update hardcoded prose numbers in this file"
B8={n:json.load(open(f"{OUT}/bin8um/{n}.json")) for n in ["stdxenium_breast","wta_breast","stratamap_breast"]
    if os.path.exists(f"{OUT}/bin8um/{n}.json")}   # unit-matched 8um-bin metrics (per-gene on L1)

# ---- diffusion / lateral spillover (29_diffusion.py, 30_diffusion_figure.py, 31_grid_registration.py)
def _load(p, default=None):
    return json.load(open(p)) if os.path.exists(p) else default
DIF  = _load(f"{OUT}/diffusion/_summary.json", {})       # per platform, per compartment
DREG = _load(f"{OUT}/diffusion/_registration.json", {})  # grid-vs-tissue offset (Visium HD)
DST  = _load(f"{OUT}/diffusion/_selftest.json", {})      # estimator positive controls
DCAL = _load(f"{OUT}/diffusion/_drift_calibration.json", {})
DV1  = _load(f"{OUT}/diffusion/_v1_control.json", {})     # REAL-DATA control: Visium v1
DOTA = _load(f"{OUT}/diffusion/_offtissue_all.json", {})  # all six on the v1 off-tissue axis
DIF_BIN_UM = 8.0
V1  = {k: v for k, v in DV1.items() if isinstance(v, dict) and v.get("kind") == "v1"
       and "error" not in v}
V1H = {k: v for k, v in DV1.items() if isinstance(v, dict) and v.get("kind") == "hd"
       and "error" not in v}
def _mean(d, f):
    xs = [v[f] for v in d.values() if v.get(f) is not None]
    return (sum(xs) / len(xs)) if xs else None

# ---- platform identity ------------------------------------------------------
PLAT=["stdxenium_breast","prime5k_breast","wta_breast","visiumhd65_breast_8um","visiumhd11_breast_8um","stratamap_breast"]
NAME={"stdxenium_breast":"Xenium","prime5k_breast":"Xenium Prime 5K","wta_breast":"Atera",
      "visiumhd65_breast_8um":"Visium HD 6.5 mm","visiumhd11_breast_8um":"Visium HD 11 mm",
      "stratamap_breast":"Illumina StrataMap"}
SUB ={"stdxenium_breast":"313-plex probe · FFPE","prime5k_breast":"~5,101-plex probe · FFPE","wta_breast":"18,028-gene WTx probe · FFPE",
      "visiumhd65_breast_8um":"probe WTx, 8 µm bin · FFPE","visiumhd11_breast_8um":"probe WTx, 8 µm bin · FFPE",
      "stratamap_breast":"poly(A) WTx, 1 µm · FF"}
CLR ={"stdxenium_breast":"#1B9E77","prime5k_breast":"#D95F02","wta_breast":"#7570B3",
      "visiumhd65_breast_8um":"#E7298A","visiumhd11_breast_8um":"#66A61E","stratamap_breast":"#2E6FAF"}
MOD ={"stdxenium_breast":"imaging","prime5k_breast":"imaging","wta_breast":"imaging",
      "visiumhd65_breast_8um":"sequencing","visiumhd11_breast_8um":"sequencing","stratamap_breast":"sequencing"}
# Tissue preparation per platform (all breast columns are FFPE except StrataMap; the fresh-frozen
# Visium HD is a separate dataset discussed in the per-cell section, not a table column).
PREP={"stdxenium_breast":"FFPE","prime5k_breast":"FFPE","wta_breast":"FFPE",
      "visiumhd65_breast_8um":"FFPE","visiumhd11_breast_8um":"FFPE","stratamap_breast":"fresh-frozen"}
# Capture chemistry -> readout. Two independent axes: how transcripts are captured (probe
# hybridisation vs native poly-A) and how they are read out (in-situ imaging vs sequencing).
# Xenium/Prime/Atera = probe + imaging; Visium HD = probe + sequencing (CytAssist); StrataMap = poly-A + sequencing.
CHEM={"stdxenium_breast":"probe &rarr; imaging","prime5k_breast":"probe &rarr; imaging","wta_breast":"probe &rarr; imaging",
      "visiumhd65_breast_8um":"probe &rarr; sequencing","visiumhd11_breast_8um":"probe &rarr; sequencing","stratamap_breast":"poly(A) &rarr; sequencing"}
# Throughput axis. For the IMAGING tiers a run is bounded by how many slides the instrument holds,
# so "slides/run x per-slide area" is a real, binding number: Xenium/Prime 2x235, Atera 4x500.
# A Visium HD slide has 2 capture areas (6.5mm A1/D1, 11mm A/B) = 2x42.25 / 2x121 mm2, but is billed
# per REACTION = per single capture area (FGCZ/Christian price is per reaction, not per slide);
# StrataMap holds multiple sections in ~624 mm2 usable. For both, throughput is
# ultimately sequencing-capacity-limited, not slide-limited.
SLIDES_PER_RUN={"stdxenium_breast":2,"prime5k_breast":2,"wta_breast":4,
                "visiumhd65_breast_8um":"2 areas","visiumhd11_breast_8um":"2 areas","stratamap_breast":"multi-section"}
# StrataMap usable sample area is 13 x 48 = 624 mm2 (not the full 7.5 cm2 flow cell), and its SBC
# spatial registration mandates strict >1 mm gaps between samples and to borders, so much of that
# strip is blank in practice -> real per-area cost near Atera (FGCZ wet-lab). Xenium/Atera are imaging with
# NO strict inter-sample registration limit, so they pack samples tightly and their per-area figures hold up.
MAX_AREA={"stdxenium_breast":470.0,"prime5k_breast":470.0,"wta_breast":2000.0,
          "visiumhd65_breast_8um":84.5,"visiumhd11_breast_8um":242.0,"stratamap_breast":624.0}

# ---- cost model: FGCZ all-in CUSTOMER price (CHF), i.e. what a customer pays for one billing UNIT
# INCLUDING library prep, sequencing and processing (imaging platforms need no sequencing). All six
# columns are now all-in and comparable. price = all-in price for one billing UNIT;
# area = max capture area of that unit (mm2); cost per area / per transcript at FULL utilisation.
COST={
 "stdxenium_breast":   dict(price=13321, cur="CHF", unit="run (2 slides)", area=470.0, src="FGCZ",
                            note="all-in customer price (incl. sequencing + processing)"),
 "prime5k_breast":     dict(price=21782, cur="CHF", unit="run (2 slides)", area=470.0, src="FGCZ",
                            note="all-in customer price (incl. sequencing + processing)"),
 "wta_breast":         dict(price=29400, cur="CHF", unit="run (2 slides)", area=1000.0, src="FGCZ",
                            note="all-in customer price (imaging run + processing; ~29,400 CHF / 2-slide run)"),
 "visiumhd65_breast_8um": dict(price=3862, cur="CHF", unit="1 reaction (1 capture area, 6.5 mm)", area=42.25, src="FGCZ",
                            note="all-in customer price per REACTION (incl. sequencing + processing); 1 reaction = 1 capture area = 42.25 mm2 (a slide carries 2)"),
 "visiumhd11_breast_8um": dict(price=5735, cur="CHF", unit="1 reaction (1 capture area, 11 mm)", area=121.0, src="FGCZ",
                            note="all-in customer price per REACTION (incl. sequencing + processing); 1 reaction = 1 capture area = 121 mm2 (a slide carries 2)"),
 "stratamap_breast":   dict(price=9385, cur="CHF", unit="1 slide", area=624.0, src="FGCZ",
                            note="all-in customer price; usable sample area 13x48=624 mm2 (not the full 7.5 cm2 flow cell)"),
}
def cost_per_mm2(k):
    p=COST[k]["price"]; return (p/COST[k]["area"]) if p else None
def cost_per_100k(k):
    p=COST[k]["price"]; return (p/(D[k]["transcripts_per_mm2"]*COST[k]["area"]/1e5)) if p else None

# ---- mean concordance per dataset (off-diagonal, L1) -------------------------
def mean_concord():
    rows=list(csv.reader(open(f"{OUT}/concordance_L1.csv"))); labs=rows[0][1:]
    M=np.array([[float(x) for x in r[1:]] for r in rows[1:]])
    out={}
    for i,l in enumerate(labs):
        off=[M[i,j] for j in range(len(labs)) if j!=i]
        out[l]=float(np.mean(off))
    return out
CONC=mean_concord()

# ---- cost-sensitivity figure (FGCZ all-in customer prices; CHF / mm2 vs utilisation) -
fig,ax=plt.subplots(figsize=(7.8,4.6))
u=np.linspace(0.1,1.0,50)
for k in ["visiumhd65_breast_8um","visiumhd11_breast_8um","stratamap_breast"]:
    ax.plot(u*100,COST[k]["price"]/(COST[k]["area"]*u),color=CLR[k],lw=2.4,label=f"{NAME[k]} (seq)")
for k in ["stdxenium_breast","prime5k_breast","wta_breast"]:
    ax.axhline(cost_per_mm2(k),color=CLR[k],lw=2.0,ls="--",label=f"{NAME[k]} (imaging)")
ax.set_yscale("log"); ax.set_xlabel("capture-area utilisation (% of tissue covered)")
ax.set_ylabel("all-in cost  (CHF / mm² tissue, log)")
ax.set_title("All-in cost per area: imaging vs sequencing, and the utilisation lever",fontweight="bold",fontsize=12)
ax.legend(fontsize=8.2,ncol=2,loc="upper right"); ax.grid(True,alpha=0.25)
fig.tight_layout(); fig.savefig(f"{FIG}/10_cost_sensitivity.png",bbox_inches="tight",dpi=140); plt.close(fig)

def b64(name):
    with open(f"{FIG}/{name}.png","rb") as f: return "data:image/png;base64,"+base64.b64encode(f.read()).decode()

# =============================================================== HTML helpers
def fmt(v,kind="num"):
    if v is None or (isinstance(v,float) and v!=v): return "n/a"
    if kind=="int": return f"{v:,.0f}"
    if kind=="money": return f"${v:,.0f}"
    if kind=="money1": return f"${v:,.1f}"
    if kind=="chf": return f"{v:,.0f}"
    if kind=="chf1": return f"{v:,.1f}"
    if kind=="chftx": return f"{v:,.0f}" if v>=10 else f"{v:.2g}"   # 2 sig figs so tiny values don't read as 0.0
    if kind=="pct3": return f"{v*100:.3f}%"
    if kind=="pct2": return f"{v*100:.2f}%"
    if kind=="num2": return f"{v:.2f}"
    if kind=="g2": return f"{v:,.0f}" if v>=100 else f"{v:.3g}"
    return f"{v:,.0f}"

def chip(k):
    return (f'<span class="chip"><span class="dot" style="background:{CLR[k]}"></span>'
            f'{NAME[k]}</span>')

# ---- master table -----------------------------------------------------------
def th_platforms():
    cells="".join(
        f'<th class="pcol" style="--pc:{CLR[k]}"><span class="pname">{NAME[k]}</span>'
        f'<span class="psub">{SUB[k]}</span></th>' for k in PLAT)
    return f'<tr><th class="rlab"></th>{cells}</tr>'

def row(label, fn, kind="num", note=None, best="hi"):
    vals=[fn(k) for k in PLAT]
    nums=[v for v in vals if isinstance(v,(int,float)) and v==v]
    bestv=(max(nums) if best=="hi" else min(nums)) if nums else None
    tds=""
    for k,v in zip(PLAT,vals):
        cls="num"
        if bestv is not None and isinstance(v,(int,float)) and v==v and abs(v-bestv)<1e-9 and best in("hi","lo"):
            cls="num best"
        tds+=f'<td class="{cls}">{fmt(v,kind)}</td>'
    n=f'<span class="rnote">{note}</span>' if note else ""
    return f'<tr><td class="rlab">{label}{n}</td>{tds}</tr>'

def grp(t): return f'<tr class="grp"><td class="rlab">{t}</td>'+("<td></td>"*len(PLAT))+"</tr>"

def textrow(label,fn,note=None):
    tds="".join(f'<td class="num">{fn(k)}</td>' for k in PLAT)
    n=f'<span class="rnote">{note}</span>' if note else ""
    return f'<tr><td class="rlab">{label}{n}</td>{tds}</tr>'

table = '<div class="tablewrap"><table class="master">'
table += "<thead>"+th_platforms()+"</thead><tbody>"
table += grp("Design")
table += textrow("Modality", lambda k: MOD[k].title())
table += textrow("Tissue prep", lambda k: (f'<b style="color:#B00">{PREP[k]}</b>' if PREP[k]!="FFPE" else PREP[k]),
                 note="all breast columns are FFPE except StrataMap (fresh-frozen); a fresh-frozen Visium HD is compared separately in the per-cell section")
table += textrow("Capture chemistry &rarr; readout", lambda k: CHEM[k],
                 note="two axes: capture (probe hybridisation vs native poly-A) then readout (imaging vs sequencing). Xenium/Prime/Atera hybridise gene-specific probes read out by in-situ imaging; Visium HD hybridises probes (CytAssist Transcriptome set) then sequences the library; StrataMap captures poly-A transcripts then sequences")
table += row("Gene-expression genes", lambda k:D[k]["n_gex_genes"], "int", note="panel size; StrataMap bounded to protein-coding detected (+~31k noncoding features it also captures)", best="hi")
table += textrow("Slides / capture areas per run", lambda k: f"{SLIDES_PER_RUN[k]}",
                 note="samples one instrument run processes at once - a binding throughput number for the imaging tiers; for Visium HD and StrataMap (sequencing readout) throughput is sequencing-capacity-limited, not slide-limited (a Visium HD slide carries 2 capture areas; a StrataMap slide holds multiple sections)")
table += row("Max area per run (mm²)", lambda k:MAX_AREA[k], "g2", note="imaging tiers = slides/run × per-slide area (Atera 4×500, Xenium/Prime 2×235) and this is binding; the sequencing columns show capture format only (Visium HD 2×42/121, StrataMap 624/slide) - their real throughput is sequencing-limited, not area-limited", best="hi")
table += row("Cells / bins analysed", lambda k:(D[k].get("percell_n_cells_detected") if k=="stratamap_breast" else D[k]["n_units"]), "int", note="segmented cells (imaging, StrataMap) or 8 µm bins (Visium HD); StrataMap also has 44.8 M raw 1 µm features", best="none")
table += textrow("Spatial unit", lambda k: ("1 µm feature" if k=="stratamap_breast" else ("native cell" if MOD[k]=="imaging" else "8 µm bin")),
                 note="Visium HD 11 mm also has SpaceRanger 4.1 cell segmentation; StrataMap per-cell rows = its 1 µm features aggregated into the demo's segmentation contours (718k cells)")
table += grp("Sensitivity &mdash; full panel")
table += row("Median transcripts / unit", lambda k:D[k]["median_transcripts_per_cell"], "int", best="hi")
table += row("Median genes / unit", lambda k:D[k]["median_genes_per_cell"], "int", best="hi")
table += row("Transcripts / mm² (total)", lambda k:D[k]["transcripts_per_mm2"], "g2", best="hi")
table += grp("Sensitivity &mdash; flagship-panel core (apples-to-apples)")
table += row("Transcripts / mm² / gene &middot; 304 genes", lambda k:D[k].get("L1_transcripts_per_mm2_per_gene"), "g2", note="per-gene depth on the 313-plex panel genes the whole-transcriptome comparators share; Prime 5K, a targeted panel, is shown on the 187 of them it carries", best="hi")
table += row("Median shared-gene tx / unit &middot; 304 genes", lambda k:(None if k=="prime5k_breast" else D[k].get("L1_median_transcripts_per_cell")), "int", note="a per-unit SUM, so a platform missing core genes scores low for that reason alone; Prime 5K carries only 187 of the 304, so it is n/a here (not comparable) - its depth is in the per-gene row above and on the L2 track, not missing", best="hi")
table += grp("Specificity")
table += row("Neg-control-probe rate (% of GEX)", lambda k:D[k].get("negctrl_probe_frac_of_gex"), "pct3", note="imaging only; raw share of counts on neg-control probes. Inflated for small panels - Xenium 313 spends ~8% of its probes on controls vs &lt;1% on 5K/Atera - so NOT a cross-panel specificity number on its own, hence no &#9733; here (the fair per-probe row below carries it)", best="none")
table += row("Neg-control per-probe background (% of mean gene)", lambda k:D[k].get("negctrl_norm_rate"), "pct2", note="imaging only; per-control-probe counts relative to the mean gene - removes the panel-size confound. This is the fair specificity comparison: ~6.5&times; Xenium vs 5K/Atera, not the ~88&times; the raw row suggests", best="lo")
table += row("Cross-platform concordance (mean &rho;)", lambda k:CONC.get(k), "num2", best="hi")
table += grp("Cost (FGCZ all-in customer price, CHF)")
table += textrow("Sequencing in price?", lambda k: "n/a (imaging)" if MOD[k]=="imaging" else "yes, included")
table += textrow("All-in price", lambda k: ("n/a" if COST[k]["price"] is None else f'{COST[k]["price"]:,.0f} {COST[k]["cur"]}' + (" *" if COST[k]["src"]=="est" else "")),
                 note="all-in customer prices incl. library prep, sequencing and processing (imaging platforms need no sequencing)")
table += textrow("Billing unit", lambda k: COST[k]["unit"])
table += row("CHF / mm² (at full capacity)", lambda k:cost_per_mm2(k), "chf", note="all-in price ÷ max capture area, at full utilisation (best case)", best="lo")
table += row("CHF / 100k transcripts", lambda k:cost_per_100k(k), "chftx", note="all-in price per 100,000 transcripts at full utilisation. Read across platforms with care: a “transcript” differs by method - StrataMap counts 1 µm spatial-barcode features, imaging counts decoded molecules, Visium HD counts UMIs - so the densest counter (StrataMap) looks cheapest partly by definition", best="lo")
table += "</tbody></table></div>"

# ---- figure section helper --------------------------------------------------
def figblock(fid,name,title,body):
    return (f'<section class="fblock" id="{fid}"><div class="ftext"><h3>{title}</h3>{body}</div>'
            f'<figure><img src="{b64(name)}" alt="{title}"/></figure></section>')

# ---- application matrix ------------------------------------------------------
APP=[
 ("Rare / low-RNA cell detection (T cells, etc.)",
  {"stdxenium_breast":"best","prime5k_breast":"ok","wta_breast":"good","visiumhd65_breast_8um":"poor","visiumhd11_breast_8um":"poor","stratamap_breast":"good"}),
 ("Whole-transcriptome discovery (unbiased)",
  {"stdxenium_breast":"no","prime5k_breast":"ok","wta_breast":"good","visiumhd65_breast_8um":"good","visiumhd11_breast_8um":"good","stratamap_breast":"best"}),
 ("Large area / many samples on a budget",
  {"stdxenium_breast":"best","prime5k_breast":"good","wta_breast":"best","visiumhd65_breast_8um":"poor","visiumhd11_breast_8um":"ok","stratamap_breast":"best"}),
 ("Subcellular / single-molecule resolution",
  {"stdxenium_breast":"best","prime5k_breast":"best","wta_breast":"best","visiumhd65_breast_8um":"ok","visiumhd11_breast_8um":"ok","stratamap_breast":"good"}),
 ("FFPE tissue",
  {"stdxenium_breast":"best","prime5k_breast":"best","wta_breast":"best","visiumhd65_breast_8um":"good","visiumhd11_breast_8um":"good","stratamap_breast":"no"}),
 ("Cost per transcript (yield efficiency)",
  {"stdxenium_breast":"good","prime5k_breast":"poor","wta_breast":"best","visiumhd65_breast_8um":"ok","visiumhd11_breast_8um":"good","stratamap_breast":"best"}),
]
RANK={"best":("Best","r-best"),"good":("Good","r-good"),"ok":("OK","r-ok"),"poor":("Limited","r-poor"),"no":("No","r-no")}
def appmatrix():
    head="".join(f'<th class="pcol" style="--pc:{CLR[k]}"><span class="pname">{NAME[k].replace(" 6.5 mm"," 6.5").replace(" 11 mm"," 11")}</span></th>' for k in PLAT)
    body=""
    for app,d in APP:
        cells=""
        for k in PLAT:
            lab,cls=RANK[d[k]]
            cells+=f'<td class="rank {cls}">{lab}</td>'
        body+=f'<tr><td class="rlab">{app}</td>{cells}</tr>'
    return f'<div class="tablewrap"><table class="master appm"><thead><tr><th class="rlab">Application</th>{head}</tr></thead><tbody>{body}</tbody></table></div>'

# =============================================================== CSS (raw)
CSS=r"""
<style>
:root{
 --ink:#0E1B22; --paper:#F4F6F7; --surface:#FFFFFF; --muted:#5B6B72; --faint:#8A9AA0;
 --line:#DBE3E6; --accent:#0C7C86; --accent2:#0A5A62; --best:#EAF6F0;
 --x1:#1B9E77; --x2:#D95F02; --x3:#7570B3; --h1:#E7298A; --h2:#66A61E;
 --mono:ui-monospace,"SFMono-Regular","Cascadia Code","JetBrains Mono",Menlo,Consolas,monospace;
 --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
*{box-sizing:border-box}
body{margin:0}
.page{background:var(--paper);color:var(--ink);font-family:var(--sans);line-height:1.55;
 -webkit-font-smoothing:antialiased;font-size:16px}
.wrap{max-width:1080px;margin:0 auto;padding:0 24px}
h1,h2,h3{text-wrap:balance;line-height:1.15;margin:0}
a{color:var(--accent2)}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:var(--accent2)}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}

/* masthead */
.mast{background:linear-gradient(180deg,#0E1B22 0%,#10242C 100%);color:#EaF2F3;padding:54px 0 40px}
.mast .wrap{max-width:1080px}
.mast .eyebrow{color:#5FD0D8}
.mast h1{font-size:clamp(30px,4.4vw,50px);font-weight:760;letter-spacing:-.015em;margin:.35em 0 .35em;color:#fff}
.mast h1 b{color:#7DE0E7;font-weight:760}
.lede{font-size:clamp(16px,1.7vw,19px);color:#B7C7CC;max-width:64ch}
.legend{display:flex;flex-wrap:wrap;gap:8px 16px;margin-top:24px}
.chip{display:inline-flex;align-items:center;gap:8px;font-family:var(--mono);font-size:12.5px;
 color:#D6E3E6;background:#ffffff10;border:1px solid #ffffff22;border-radius:999px;padding:5px 11px}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block}
.meta{display:flex;flex-wrap:wrap;gap:18px;margin-top:22px;font-family:var(--mono);font-size:12px;color:#8FA6AC}
.meta b{color:#CFE0E3;font-weight:600}

/* verdict */
.verdict{background:var(--surface);border:1px solid var(--line);border-left:4px solid var(--accent);
 border-radius:12px;padding:22px 24px;margin:-26px auto 0;position:relative;max-width:1032px;
 box-shadow:0 10px 30px -18px #0E1B2255}
.verdict p{margin:0;font-size:17px}
.verdict .k{font-weight:680;color:var(--accent2)}

section.band{padding:46px 0}
section.band h2{font-size:clamp(22px,2.6vw,30px);font-weight:720;letter-spacing:-.01em}
section.band .sub{color:var(--muted);max-width:70ch;margin:.5em 0 0}
.kicker{display:flex;align-items:baseline;gap:12px;margin-bottom:22px}
.kicker .n{font-family:var(--mono);font-size:13px;color:var(--faint)}

/* table */
.tablewrap{border:1px solid var(--line);border-radius:12px;background:var(--surface)}
@media (max-width:1080px){.tablewrap{overflow-x:auto}}   /* horizontal scroll only when the table can't fit */
table.master{border-collapse:collapse;width:100%;min-width:760px;font-size:14px}
table.master th,table.master td{padding:11px 13px;text-align:right;border-bottom:1px solid var(--line)}
/* header row stays pinned to the top of the viewport while scrolling the page */
table.master thead th{position:sticky;top:0;background:var(--surface);z-index:2;vertical-align:bottom;
 border-bottom:2px solid var(--ink);box-shadow:0 6px 14px -10px rgba(16,42,50,.40)}
table.master thead th.rlab{z-index:3}   /* top-left corner above both sticky axes */
.pcol{border-top:3px solid var(--pc)}
.pname{display:block;font-weight:700;font-size:13.5px}
.psub{display:block;font-family:var(--mono);font-size:10.5px;color:var(--muted);font-weight:400;margin-top:2px}
.rlab{text-align:left;font-weight:560;color:var(--ink);min-width:230px;position:sticky;left:0;background:var(--surface);z-index:1;border-right:1px solid var(--line)}
.rnote{display:block;font-weight:400;font-size:11.5px;color:var(--faint);margin-top:1px;max-width:30ch}
td.num{font-family:var(--mono);font-variant-numeric:tabular-nums;color:#23343b}
tr.grp td{background:#EEF3F4;font-family:var(--mono);font-size:11px;letter-spacing:.12em;text-transform:uppercase;
 color:var(--accent2);font-weight:700;border-bottom:1px solid var(--line);padding:7px 13px}
td.num.best{background:var(--best);color:#0A5A62;font-weight:700;border-radius:0}
td.num.best::after{content:" \2605";color:var(--accent);font-size:10px}
/* zebra striping (row-level, so best/group/label cells keep their own fill) */
table.master:not(.appm) tbody tr:nth-child(even){background:#F5F9F9}
table.master:not(.appm) tbody tr:nth-child(even) td.rlab{background:#F5F9F9}
table.master:not(.appm) tbody tr:hover td{background:#EAF3F3}
table.master:not(.appm) tbody tr:hover td.rlab{background:#EAF3F3}
.appm tbody tr:hover td.rank{filter:brightness(1.06)}

/* figure blocks */
.fblock{display:grid;grid-template-columns:0.82fr 1.18fr;gap:30px;align-items:center;
 background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:24px;margin-top:18px}
.fblock:nth-child(even){grid-template-columns:1.18fr 0.82fr}
.fblock:nth-child(even) .ftext{order:2}
.fblock h3{font-size:18px;font-weight:680;margin-bottom:.4em}
.fblock p{color:var(--muted);font-size:14.5px;margin:.5em 0 0}
.fblock figure{margin:0}
.fblock img{width:100%;height:auto;border-radius:8px;display:block}
.stat{font-family:var(--mono);color:var(--accent2);font-weight:700}

/* application matrix */
.appm td.rank{font-family:var(--mono);font-size:12px;font-weight:700;text-align:center;color:#fff}
.r-best{background:#0C7C86}.r-good{background:#5FAE8E}.r-ok{background:#C9B45E;color:#2a2a16}
.r-poor{background:#E3A08C;color:#3a1d12}.r-no{background:#EBEEEF;color:#8A9AA0}
.appm .rlab{min-width:280px}

/* cards / callouts */
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:8px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:20px}
.card .big{font-family:var(--mono);font-size:30px;font-weight:740;color:var(--accent2);letter-spacing:-.02em}
.card h4{margin:.1em 0 .3em;font-size:14px}
.card p{margin:0;color:var(--muted);font-size:13px}
.note{font-size:12.5px;color:var(--faint)}

details.methods{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:6px 22px;margin-top:18px}
details.methods summary{cursor:pointer;font-weight:660;padding:14px 0;font-size:15px}
details.methods[open] summary{border-bottom:1px solid var(--line);margin-bottom:14px}
details.methods h4{margin:16px 0 6px;font-size:13px;font-family:var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--accent2)}
details.methods p,details.methods li{font-size:13.5px;color:#374a51}
details.methods code{font-family:var(--mono);font-size:12px;background:#EEF3F4;padding:1px 5px;border-radius:4px}

footer{padding:40px 0 60px;color:var(--faint);font-size:12.5px;font-family:var(--mono)}
@media (max-width:760px){
 .fblock,.fblock:nth-child(even){grid-template-columns:1fr}
 .fblock:nth-child(even) .ftext{order:0}
 .grid3{grid-template-columns:1fr}
 .rlab{position:static}
}
:focus-visible{outline:2.5px solid var(--accent);outline-offset:2px}
@media (prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
</style>
"""

# =============================================================== compute headline numbers for prose
std=D["stdxenium_breast"]; wta=D["wta_breast"]; hd=D["visiumhd65_breast_8um"]; p5=D["prime5k_breast"]
ratio_std_hd = std["L1_transcripts_per_mm2_per_gene"]/hd["L1_transcripts_per_mm2_per_gene"]
cpmm_hd=cost_per_mm2("visiumhd65_breast_8um"); cpmm_std=cost_per_mm2("stdxenium_breast")
c100_wta=cost_per_100k("wta_breast"); c100_hd=cost_per_100k("visiumhd65_breast_8um")

# ---- Atera vs CosMx WTx vignette numbers (cross-vendor whole-transcriptome imaging) ----
cos=json.load(open(f"{OUT}/per_dataset/cosmx_breast.json"))
_ate=json.load(open(f"{OUT}/per_dataset/wta_breast.json"))
_pgA=_ate["total_gex_transcripts"]/_ate["n_gex_genes"]/_ate["n_units"]
atera_cw_norm=(_ate["negctrl_codeword_counts"]/_ate["n_negctrl_codewords"]/_ate["n_units"])/_pgA
cx_depth_g=_ate["median_genes_per_cell"]/cos["median_genes_per_cell"]
cx_pergene=cos["shared_atera_mean_tx_per_gene_per_cell_atera"]/cos["shared_atera_mean_tx_per_gene_per_cell_cosmx"]
cx_bg_neg=cos["negctrl_norm_rate"]/_ate["negctrl_norm_rate"]
cx_bg_cw=cos["falsecode_norm_rate"]/atera_cw_norm
cx_shared=cos["shared_with_atera_n_genes"]; cx_conc=cos["concordance_spearman_vs_atera"]

# ---- Illumina StrataMap: recomputed, segmentation-free (raw 1um-SBC matrix) ----
sm=json.load(open(f"{OUT}/per_dataset/stratamap_breast.json"))
sm_vs_xen=sm["L1_transcripts_per_mm2_per_gene"]/std["L1_transcripts_per_mm2_per_gene"]   # vs the prev per-gene leader
sm_vs_atera_genes=sm["n_gex_genes"]/wta["n_gex_genes"]

# =============================================================== BODY
B=[]
B.append('<div class="page">')

# masthead
B.append('<header class="mast"><div class="wrap">')
B.append('<div class="eyebrow">Spatial transcriptomics &middot; multi-vendor benchmark on matched human breast cancer</div>')
B.append('<h1>Xenium, <b>Atera</b>, Visium HD &amp; <b>StrataMap</b>,<br>measured on the same genes</h1>')
B.append('<p class="lede">Six spatial assays across three vendors - 10x&rsquo;s imaging tiers (313-plex &rarr; 5K &rarr; '
         'whole-transcriptome Atera) and Visium HD sequencing, Illumina&rsquo;s new poly(A) whole-transcriptome '
         '<b>StrataMap</b>, and a CosMx cross-check - each recomputed from raw count matrices on a shared gene set, '
         'so sensitivity, specificity and price-per-area are finally comparable.</p>')
B.append('<div class="legend">'+''.join(chip(k) for k in PLAT)+'</div>')
B.append('<div class="meta"><span>tissue&nbsp; <b>breast cancer FFPE</b> (+ cervix, 2 tiers)</span>'
         '<span>method&nbsp; <b>uniform recompute</b>, not vendor summaries</span>'
         '<span>shared sets&nbsp; <b>304</b> / <b>4,843</b> genes</span></div>')
B.append('</div></header>')

# verdict
B.append('<div class="wrap"><div class="verdict"><p><span class="k">Bottom line.</span> '
         'For <b>FFPE</b> there is no single winner - a <b>breadth&ndash;depth&ndash;cost triangle</b>: '
         f'targeted Xenium is ~<span class="mono">{ratio_std_hd:.0f}&times;</span> more sensitive per gene than Visium HD and '
         f'~<span class="mono">{cpmm_hd/cpmm_std:.0f}&times;</span> cheaper per mm² than Visium HD 6.5&nbsp;mm (all-in) but sees only a curated panel; '
         'Atera adds the whole transcriptome at imaging-grade sensitivity and the lowest cost per transcript among the FFPE options; '
         'Visium HD is the unbiased, lowest-capital option but pays in per-gene sensitivity. '
         'On <b>fresh-frozen</b> the triangle bends: Illumina&rsquo;s <b>StrataMap</b> posts the highest per-gene sensitivity '
         f'here (<span class="mono">{sm["L1_transcripts_per_mm2_per_gene"]:,.0f}</span> tx/mm²/gene) across the whole '
         'transcriptome, and is the <b>cheapest all-in per transcript</b> (per area it is also lowest on paper, though gaps between samples make that best-case) - but it is fresh-frozen only, so it does not replace the FFPE options.</p></div></div>')

# section: master table
B.append('<section class="band"><div class="wrap">')
B.append('<div class="kicker"><span class="n">01</span><h2>The comparison, one screen</h2></div>')
B.append('<p class="sub" style="margin-bottom:18px">Every value recomputed from the count matrix with one definition. '
         '&#9733; marks the best platform in each row. Imaging tiers use native segmented cells, Visium HD 8&nbsp;µm bins, '
         'StrataMap its raw 1&nbsp;µm features - so <b>per-area</b> and <b>per-gene</b> rows are the apples-to-apples ones; '
         'per-cell rows are n/a where the native unit is not a cell.</p>')
B.append(table)
B.append('<div class="card" style="border-left:4px solid var(--faint);margin-top:18px;background:#FbFcFc">'
 '<h4 style="margin:.1em 0 .4em;font-size:15px">Not benchmarked here: Singular Genomics <b>G4X</b> (no public data)</h4>'
 '<p style="color:var(--muted);font-size:14px;margin:0">The G4X spatial sequencer is a <b>targeted in-situ multiomic</b> platform - a '
 '<b>300-plex</b> standard breast panel (custom up to 500-plex), on <b>FFPE</b>, reading RNA (Direct-Seq) plus an 18-plex protein layer '
 'and fluorescent H&amp;E at subcellular resolution. By panel size it sits in the <b>Xenium 313-plex tier</b> (with added protein/H&amp;E). '
 'It is left out of the table above on purpose: <b>no public breast-cancer count matrix exists</b> (Singular releases data only via request '
 'forms), so it cannot be uniformly recomputed like every column here, and vendor summary numbers are not apples-to-apples with the rest. '
 'For scale only (vendor spec, <em>different tissue</em>, not recomputed): a 300-plex run on FFPE <em>kidney</em> cancer reported '
 '~<b>78</b> transcripts/cell (6.2 M cells, 483 M transcripts across 10 sections) - the same order as a targeted panel here. '
 'If a G4X breast matrix becomes available it can be dropped straight into this benchmark.</p></div>')
B.append('</div></section>')

# section 02: sensitivity / breadth-depth
B.append('<section class="band" style="background:#fff;border-top:1px solid var(--line);border-bottom:1px solid var(--line)"><div class="wrap">')
B.append('<div class="kicker"><span class="n">02</span><h2>Sensitivity: breadth costs per-gene depth</h2></div>')
B.append(figblock("f-pergene","01_pergene_sensitivity_L1","Per-gene detection power, on the 304 shared genes",
   f'<p>On the shared genes, the targeted 313-plex Xenium records '
   f'<span class="stat">{std["L1_transcripts_per_mm2_per_gene"]:.0f}</span> transcripts/mm²/gene - about '
   f'<span class="stat">{ratio_std_hd:.0f}&times;</span> a Visium HD 8&nbsp;µm bin and '
   f'<span class="stat">{std["L1_transcripts_per_mm2_per_gene"]/p5["L1_transcripts_per_mm2_per_gene"]:.0f}&times;</span> Xenium Prime 5K: '
   f'fewer targets means more probes per gene and deeper signal. The fresh-frozen <b>StrataMap</b> tops even that '
   f'(<span class="stat">{sm["L1_transcripts_per_mm2_per_gene"]:,.0f}</span>) - whole-transcriptome, but see its fresh-frozen caveat below.</p>'))
B.append(figblock("f-bd","04_breadth_vs_depth","The breadth&ndash;depth frontier",
   '<p>Plotting breadth against per-gene sensitivity exposes the trade-off curve. '
   'The 313-plex panel sits top-left (deep but narrow); Visium HD and Prime 5K sit bottom-right (broad but shallow); '
   '<b>Atera</b> bends it (whole-transcriptome breadth at near-targeted depth). <b>StrataMap</b> sits alone <b>top-right</b> - '
   'highest per-gene depth <em>and</em> whole-transcriptome breadth - because unbiased fresh-frozen sequencing is not spread across a '
   'fixed panel. On FFPE, though, the trade-off still binds (see the StrataMap caveats).</p>'))
B.append(figblock("f-yield","02_fullpanel_tx_per_cell","Total molecular yield per cell (protein-coding)",
   f'<p>On <b>protein-coding</b> genes - the fair basis, since every other platform is a protein-coding-oriented panel - imaging '
   f'<b>Atera</b> captures a median <span class="stat">{wta["median_transcripts_per_cell"]:.0f}</span> transcripts per cell, an order of '
   f'magnitude past the other FFPE tiers and approaching dissociated single-cell depth in situ. <b>StrataMap</b> is higher still '
   f'(<span class="stat">{sm.get("median_transcripts_per_cell_pc", sm["median_transcripts_per_cell"]):,.0f}</span> protein-coding transcripts per cell, '
   'its 1 µm features aggregated into the demo&rsquo;s 718k segmentation cells) - but it is <b>fresh-frozen</b>, so that edge is '
   'partly prep, not just platform (see the fresh-frozen callout below). '
   f'<span class="note">StrataMap is whole-transcriptome: counting all {sm["n_gex_genes_annotation"]:,} GENCODE features (incl. ~{sm["n_noncoding_detected"]//1000}k detected '
   f'noncoding genes) raises this only to {sm["median_transcripts_per_cell"]:,.0f} (+6%) - many noncoding genes, but few transcripts each - so the '
   'protein-coding number is the honest one for comparison.</span></p>'))
B.append(figblock("f-percell-shared","15_percell_sharedgene","Per-cell depth on specific genes: the counterpoint to per-area",
   f'<p>Flip the axis from per-mm² to <em>per cell, on the 304 shared marker genes</em>, and the ranking inverts. '
   f'A targeted 313-plex Xenium cell holds <span class="stat">{std["L1_median_transcripts_per_cell"]:.0f}</span> shared-gene transcripts, '
   f'vs Atera <span class="stat">{wta["L1_median_transcripts_per_cell"]:.0f}</span> and '
   f'<span class="stat">{sm["L1_median_transcripts_per_cell"]:.0f}</span> for StrataMap - the whole-transcriptome platforms that '
   '<em>led</em> the per-mm² chart now trail, because a fixed panel concentrates its reads on the genes you asked for while an unbiased '
   'assay spreads them across the transcriptome. Fresh-frozen prep lifts everything (Visium HD FF cell tops it), but among matched FFPE the '
   'targeted panel wins per cell. So StrataMap&rsquo;s per-area dominance and its modest per-cell marker depth are two true faces of the '
   'same breadth-vs-depth trade-off.</p>'))
# segmented-cell vs 8um-bin callout (the fair cell-vs-cell comparison)
_seg=D["visiumhd11_breast_seg"]; _b11=D["visiumhd11_breast_8um"]
_rc=std["L1_median_transcripts_per_cell"]/_seg["L1_median_transcripts_per_cell"]
_rb=std["L1_median_transcripts_per_cell"]/_b11["L1_median_transcripts_per_cell"]
B.append('<div class="card" style="border-left:4px solid var(--accent);margin-top:18px">'
 '<h4 style="margin:.1em 0 .4em;font-size:15px">A fairer per-cell read: Visium HD with real cell segmentation</h4>'
 f'<p style="color:var(--muted);font-size:14.5px;margin:0">The 11&nbsp;mm run ships SpaceRanger&nbsp;4.1 cell segmentation, '
 f'so it can be read as true cells rather than 8&nbsp;µm bins. A segmented cell holds a median '
 f'<span class="stat">{_seg["median_transcripts_per_cell"]:.0f}</span> transcripts and '
 f'<span class="stat">{_seg["median_genes_per_cell"]:.0f}</span> genes (~4&times; an 8&nbsp;µm bin), and '
 f'<span class="stat">{_seg["L1_median_transcripts_per_cell"]:.0f}</span> on the 304 shared genes (vs '
 f'{_b11["L1_median_transcripts_per_cell"]:.0f} per bin). Even cell-to-cell, the targeted 313-plex Xenium '
 f'(<span class="stat">{std["L1_median_transcripts_per_cell"]:.0f}</span>) stays ~<span class="stat">{_rc:.0f}&times;</span> '
 f'more sensitive per shared gene - the gap narrows from ~{_rb:.0f}&times; (vs bins) but does not close. '
 'The 6.5&nbsp;mm public release is binned-only; <b>per-area</b> sensitivity and cost are unaffected by the unit choice.</p></div>')
# fresh-frozen probe-based Visium HD - the prep effect
_ff=json.load(open(f"{OUT}/per_dataset/visiumhd65ff_breast_seg.json"))
_pe=_ff["L1_median_transcripts_per_cell"]/_seg["L1_median_transcripts_per_cell"]
B.append('<div class="card" style="border-left:4px solid var(--h2);margin-top:14px">'
 '<h4 style="margin:.1em 0 .4em;font-size:15px">Prep beats platform: fresh-frozen Visium HD is the most sensitive per cell</h4>'
 f'<p style="color:var(--muted);font-size:14.5px;margin:0">Swapping FFPE for a <b>fresh-frozen</b> breast block '
 f'(same probe chemistry - Human Transcriptome Probe Set v2, ~18k genes - and SpaceRanger&nbsp;4.0.1 segmentation) lifts a '
 f'Visium HD cell to a median <span class="stat">{_ff["median_transcripts_per_cell"]:,.0f}</span> transcripts / '
 f'<span class="stat">{_ff["median_genes_per_cell"]:,.0f}</span> genes, and '
 f'<span class="stat">{_ff["L1_median_transcripts_per_cell"]:.0f}</span> on the 304 shared genes - <b>above Xenium '
 f'({std["L1_median_transcripts_per_cell"]:.0f}) and Atera ({wta["L1_median_transcripts_per_cell"]:.0f})</b>, and '
 f'~<span class="stat">{_pe:.0f}&times;</span> the FFPE Visium HD cell ({_seg["L1_median_transcripts_per_cell"]:.0f}). '
 'So Visium HD&rsquo;s per-cell gap is largely an <b>FFPE</b> effect: on fresh-frozen tissue, probe-based sequencing is the '
 f'most sensitive per cell. The other fresh-frozen platform here, Illumina <b>StrataMap</b>, sits in the same bracket '
 f'(<span class="stat">{sm.get("median_transcripts_per_cell_pc", sm["median_transcripts_per_cell"]):,.0f}</span> protein-coding transcripts per cell), '
 f'though on the 304 shared genes a StrataMap cell ({sm["L1_median_transcripts_per_cell"]:.0f}) trails this '
 f'probe-based FF Visium HD ({_ff["L1_median_transcripts_per_cell"]:.0f}), which spends its probes on fewer genes. '
 'It lifts per-cell capture well above the FFPE Visium HD cell - but that gain is specifically <b>per-cell</b>: on '
 f'<b>area-normalised per-gene</b> sensitivity the targeted Xenium panel still leads ({std["L1_transcripts_per_mm2_per_gene"]:,.0f} tx/mm²/gene on the {L1N} core), '
 'so FF is a per-cell, not per-gene-per-area, advantage. Caveats - different block, Ultima-sequenced, and fresh-frozen '
 'needs frozen tissue (the Xenium / Atera samples here are FFPE); per-area cost still carries deep sequencing.</p></div>')
B.append('</div></section>')

# section 02b: 8um-bin unit-matched (no segmentation)
B.append('<section class="band" style="background:#fff;border-top:1px solid var(--line);border-bottom:1px solid var(--line)"><div class="wrap">')
B.append('<div class="kicker"><span class="n">02b</span><h2>Same units, no segmentation: the 8 µm-bin check</h2></div>')
B.append('<p class="sub">Comparing imaging cells to Visium HD bins mixes unit size and segmentation choices. To remove that '
 'confound entirely, every imaging platform&rsquo;s <b>raw transcripts</b> (qv&ge;20) <b>and StrataMap&rsquo;s raw 1 µm spatial-barcode (SBC) matrix</b> '
 'were gridded onto an identical <b>8 µm square grid</b> - molecules-into-bins, <b>no segmentation for anyone</b>, exactly how '
 'Visium HD works.</p>')
B.append(f'<figure style="margin:18px 0 0"><img src="{b64("11_bin8um_unitmatched")}" alt="Unit-matched 8 um bin comparison" '
 'style="width:100%;border:1px solid var(--line);border-radius:12px;display:block"/></figure>')
B.append('<p class="sub" style="margin-top:14px">Among the <b>FFPE</b> platforms the story is unchanged on identical units: the '
 f'targeted 313-plex Xenium concentrates a median <span class="stat" style="color:var(--accent2);font-weight:700">{B8["stdxenium_breast"]["L1_median_tx_per_bin"]:.0f}</span> '
 f'transcripts/bin on the {L1N} shared genes (vs Atera {B8["wta_breast"]["L1_median_tx_per_bin"]:.0f}, Visium HD {D["visiumhd65_breast_8um"]["L1_median_transcripts_per_cell"]:.0f}), while <b>Atera leads breadth</b> at <span class="stat" '
 f'style="color:var(--accent2);font-weight:700">{B8["wta_breast"]["median_genes_per_bin"]:.0f}</span> genes/bin - targeted panels win per-gene depth, whole-transcriptome imaging '
 'wins breadth, even molecule-for-molecule. The fresh-frozen <b>StrataMap</b> sits <b>above everything on this grid</b> '
 f'(<span class="stat" style="color:var(--accent2);font-weight:700">{B8["stratamap_breast"]["median_genes_per_bin"]:,.0f}</span> genes/bin, '
 f'<span class="stat" style="color:var(--accent2);font-weight:700">{B8["stratamap_breast"]["L1_median_tx_per_bin"]:.0f}</span> shared-gene tx/bin), but read that with two caveats: it is '
 '<b>fresh-frozen</b> (the rest are FFPE), and sequencing counts every gene with &ge;1 UMI in a bin whereas imaging counts qv-filtered '
 'decoded molecules - so its <em>breadth</em> lead in particular is inflated by counting convention as much as by prep. '
 '<span class="note">Imaging binned from <code>transcripts.parquet</code> (std Xenium 639 M, Atera 740 M); StrataMap = its 1 µm SBC '
 'matrix gridded to 8 µm (genes/bin protein-coding).</span></p>')
B.append('</div></section>')

# section 02c: "18k genes" - how many are actually usable (per-gene abundance distribution)
GA=json.load(open(f"{OUT}/gene_abundance_stats.json"))
B.append('<section class="band"><div class="wrap">')
B.append('<div class="kicker"><span class="n">02c</span><h2>&ldquo;18,000 genes&rdquo; &ndash; how many are actually usable?</h2></div>')
B.append('<p class="sub">A fair question on any whole-transcriptome count: bulk RNA-seq usually calls only ~10&ndash;15k genes '
 '<em>usable</em>, yet these assays report ~18&ndash;19k genes &ldquo;detected&rdquo;. The gap is the definition of '
 '<b>&ldquo;detected&rdquo;</b> (&ge;1 count anywhere on the section): a long tail of genes carries almost no signal. Below, each '
 'protein-coding gene&rsquo;s total count (matrix row-sum) is sorted and <b>both axes normalised to percent</b>, so the shape of the '
 'curve - not the panel size - tells the story.</p>')
B.append(f'<figure style="margin:18px 0 0"><img src="{b64("24_gene_abundance")}" alt="Per-gene abundance distribution across platforms" '
 'style="width:100%;border:1px solid var(--line);border-radius:12px;display:block"/></figure>')
B.append('<p class="sub" style="margin-top:14px">Read the middle panel (Lorenz): the coloured dot marks how many genes carry '
 f'<b>90% of all transcripts on the section</b> (the pseudobulk row-sums, i.e. bulk-equivalent - within a single cell it is fewer still). On <b>Atera</b> that is <span class="stat">{GA["wta_breast"]["n90"]:,}</span> of '
 f'<span class="stat">{GA["wta_breast"]["detected"]:,}</span> detected genes; on <b>Visium HD 6.5 mm</b> '
 f'<span class="stat">{GA["visiumhd65_breast_8um"]["n90"]:,}</span> / {GA["visiumhd65_breast_8um"]["detected"]:,}; on '
 f'<b>StrataMap</b> (protein-coding) <span class="stat">{GA["stratamap_breast"]["n90"]:,}</span> / {GA["stratamap_breast"]["detected"]:,}. '
 'So the <b>usable</b> gene count sits right at or below bulk&rsquo;s 10&ndash;15k: the remaining ~10,000 &ldquo;detected&rdquo; genes '
 'share only the last 10% of reads - a handful of counts each spread across the whole section - and are not quantitatively reliable, '
 'least of all per cell. The targeted panels invert the picture by design: Xenium&rsquo;s 313 are curated to be high-value, so '
 f'<span class="stat">{GA["stdxenium_breast"]["n90"]}</span> of 313 already carry 90% and every gene is deeply sampled. '
 f'<b>CosMx</b> is the flattest (its 90% needs <span class="stat">{GA["cosmx_breast"]["n90"]:,}</span> genes) - but that is largely a '
 '<b>noise floor</b>, not more usable biology: its negative-control probes collect ~<b>28%</b> of a real gene&rsquo;s per-cell signal '
 '(vs ~1% on Xenium and ~0.2% on Atera; see the specificity section), so background counts smear across the tail and flatten the curve. '
 'The <b>violin (right panel)</b> makes the shapes concrete: targeted Xenium 313 sits <b>high and tight</b> (few genes, all abundant); '
 'the whole-transcriptome platforms spread down into a <b>long low tail</b> (StrataMap widest - fresh-frozen poly-A reaches the rarest '
 'transcripts); while CosMx is <b>compact and lifted off the floor</b> - the signature of counts that never reach zero because of background. '
 '<span class="note">&ldquo;Detected&rdquo; = row-sum &ge;1 over the whole section; usable-per-cell is far stricter still. This is why '
 'whole-transcriptome breadth should be read as &ldquo;captures the transcriptome&rdquo;, not &ldquo;quantifies 18k genes per cell&rdquo;.</span></p>')
# StrataMap vs bulk RNA-seq
SVB=json.load(open(f"{OUT}/stratamap_vs_bulk_stats.json"))
_bk="Bulk breast RNA-seq (HPA)"; _sm="StrataMap (poly-A in-situ)"
B.append('<h3 style="margin-top:26px">Does the poly-A capture match real bulk RNA-seq?</h3>')
B.append('<p class="sub">StrataMap is poly-A, like bulk RNA-seq - so we can ask it directly against a <b>bulk breast RNA-seq</b> reference '
 '(HPA consensus). This is exactly Hubert&rsquo;s question: if bulk &ldquo;usably&rdquo; sees ~10&ndash;15k genes, where does an in-situ poly-A assay land?</p>')
B.append(f'<figure style="margin:16px 0 0"><img src="{b64("25_stratamap_vs_bulk")}" alt="StrataMap vs bulk RNA-seq gene-abundance distribution" '
 'style="width:100%;border:1px solid var(--line);border-radius:12px;display:block"/></figure>')
B.append('<p class="sub" style="margin-top:14px">The two distributions <b>largely coincide</b>. Bulk breast concentrates 90% of its signal into '
 f'<span class="stat">{SVB[_bk]["n90"]:,}</span> of {SVB[_bk]["detected"]:,} detected genes; StrataMap into '
 f'<span class="stat">{SVB[_sm]["n90"]:,}</span> of {SVB[_sm]["detected"]:,} - the same shape, StrataMap if anything spreading its signal across '
 'slightly more genes. So bulk itself obeys &ldquo;detected &raquo; usable&rdquo; (its own usable core is ~4k of ~17k), and StrataMap is a '
 'faithful <b>in-situ stand-in for bulk poly-A</b> - it captures the transcriptome with bulk-like complexity, which the probe panels (fixed ~18k '
 'protein-coding set) cannot claim. <span class="note">Unit caveat: bulk is length-normalised nTPM, StrataMap raw molecule counts, so compare curve '
 'shape not exact position; HPA breast is normal tissue. Both restricted to protein-coding.</span></p>')
# depth-matched test against a cloud of real breast tumours (TCGA-BRCA)
TCG=json.load(open(f"{OUT}/stratamap_vs_tcga_stats.json"))
B.append('<h3 style="margin-top:26px">Depth-matched: is it real breadth, or just more sequencing?</h3>')
B.append('<p class="sub">The obvious objection: StrataMap simply sequences deeper, so of course it &ldquo;sees&rdquo; more genes. To settle it we '
 f'<b>downsampled the StrataMap section to the exact depth of bulk breast tumours</b> - the median library of <span class="stat">{TCG["tcga_n_tumours"]:,}</span> '
 f'TCGA-BRCA primary tumours (<span class="stat">{TCG["tcga_median_libsize"]/1e6:.0f}M</span> protein-coding counts) - and compared it not to one reference but to the '
 '<b>whole cloud</b> of those tumours, so we can see where StrataMap falls within real biological spread.</p>')
B.append(f'<figure style="margin:16px 0 0"><img src="{b64("27_stratamap_vs_tcga")}" alt="StrataMap downsampled to bulk depth vs the TCGA-BRCA cloud" '
 'style="width:100%;border:1px solid var(--line);border-radius:12px;display:block"/></figure>')
B.append('<p class="sub" style="margin-top:14px">At matched depth StrataMap lands <b>inside the bulk cloud</b>. Its 90% of signal sits in '
 f'<span class="stat">{TCG["stratamap_downsampled_n90"]:,}</span> genes, right at the lower edge of the TCGA-BRCA band '
 f'(<span class="stat">{TCG["tcga_n90_lo"]:,}&ndash;{TCG["tcga_n90_hi"]:,}</span>, median {TCG["tcga_n90_median"]:,}), and its detected-gene count collapses from '
 f'<span class="stat">{TCG["stratamap_full_ndet"]:,}</span> at full depth to <span class="stat">{TCG["stratamap_downsampled_ndet"]:,}</span> - essentially the bulk median '
 f'(<span class="stat">{TCG["tcga_ndet_median"]:,}</span>). Two things follow. First, the extra ~2,000 genes StrataMap reports at full depth are exactly the rare tail that '
 'bulk at 56M reads would also miss - real transcripts, but not extra <em>usable</em> biology. Second, and decisively, the <b>usable core is depth-invariant</b>: '
 f'90% of signal needs <span class="stat">{TCG["stratamap_full_n90"]:,}</span> genes at full depth and <span class="stat">{TCG["stratamap_downsampled_n90"]:,}</span> at '
 'bulk depth - the same number. So &ldquo;detected &raquo; usable&rdquo; is not a StrataMap sequencing artifact; it is a genuine property of the breast transcriptome that '
 'bulk RNA-seq obeys identically. StrataMap reproduces bulk complexity at bulk depth, and keeps a real rare-gene tail beyond it. '
 '<span class="note">Downsampling is multinomial at count level (no BAM available); StrataMap counts are 1 µm feature molecules, not reads, so the depth match is on '
 'count budget. TCGA counts recovered from Xena log2(count+1); protein-coding, primary tumour only.</span></p>')
B.append('</div></section>')

# reflection: comparability of imaging vs sequencing
B.append('<section class="band"><div class="wrap">')
B.append('<div class="kicker"><span class="n">&asymp;</span><h2>How comparable are imaging and sequencing, really?</h2></div>')
B.append('<p class="sub">Imaging and sequencing count different things. Imaging (Xenium, Atera) optically <b>decodes individual '
 'probe-bound molecules in place</b> - a count is a confident molecule detection (qv-filtered). Visium HD and StrataMap <b>capture '
 'transcripts on a spatial barcode grid, then amplify and sequence</b> them - a count is a UMI, set by capture efficiency, library '
 'prep and sequencing depth (StrataMap does this at a 1 µm barcode pitch, Visium HD at 2 µm). A decoded transcript and a sequenced UMI '
 'are not the same unit, so absolute cross-modality ratios are <b>approximate, not an exact exchange rate</b>.</p>')
B.append('<p class="sub" style="margin-top:12px"><b>Segmentation is a second axis.</b> Imaging gives subcellular molecule '
 'coordinates that can be read as native <b>cells</b> or gridded into <b>bins</b>; Visium HD gives a 2 µm grid that can be binned '
 '(8 µm) or re-segmented into cells (SpaceRanger 4.x). "Cells" depends on whose segmentation and how well it draws boundaries; "bins" '
 'on bin size. Comparing imaging cells to Visium HD bins mixes both confounds - which is why this report carries per-cell, per-bin '
 '<em>and</em> per-area views plus the segmentation-free 8 µm grid above.</p>')
B.append('<div class="grid3" style="margin-top:16px">')
B.append('<div class="card" style="border-left:4px solid var(--accent)"><h4>Robust here</h4><p>Per-area / per-gene sensitivity on '
 'shared genes (intensive, segmentation-free); the 8 µm-bin grid (molecules-into-bins, no segmentation); and the '
 'breadth-vs-depth-vs-cost structure. All three tell the same story.</p></div>')
B.append('<div class="card" style="border-left:4px solid var(--x2)"><h4>Only approximate</h4><p>The exact size of any cross-modality '
 'gap - decoded-vs-UMI counting, capture efficiency, sequencing depth and FFPE-vs-fresh-frozen prep each shift it (prep alone moved '
 'Visium HD ~28&times;).</p></div>')
B.append('<div class="card" style="border-left:4px solid var(--accent2)"><h4>So read it as</h4><p>Directional, not a price list: '
 'targeted panels win per-gene depth, whole-transcriptome platforms win breadth, and sequencing depth is tunable by spend. Those '
 'hold across every view here.</p></div>')
B.append('</div></div></section>')

# section: versus dissociated single-cell
B.append('<section class="band"><div class="wrap">')
B.append('<div class="kicker"><span class="n">+</span><h2>Versus dissociated single-cell (same tissue)</h2></div>')
B.append('<p class="sub">Dissociated scRNA-seq is the conventional sensitivity reference. On breast cancer, median gene '
 "detection spans <b>1,271 to 6,966 genes/cell</b> across four chemistries - droplet 3' / 5' (poly-A) and Smart-seq2 at "
 '~1,300-1,600, up to a deeply-sequenced Flex run (probe-based, ~36k UMIs/cell) at the top - so it scales strongly with '
 'sequencing depth (CELLxGENE Census + the 10x Flex dissociated-tumour-cell set).</p>')
B.append(f'<figure style="margin:18px 0 0"><img src="{b64("12_scref_vs_spatial")}" alt="scRNA vs spatial per-cell sensitivity" '
 'style="width:100%;border:1px solid var(--line);border-radius:12px;display:block"/></figure>')
B.append('<p class="sub" style="margin-top:14px">Deeply-sequenced dissociated Flex leads at ~7,000 genes/cell - no spatial '
 'platform matches that. Among the <b>FFPE</b> spatial assays, <b>Atera</b> (1,543 genes/cell) is the standout: it reaches routine '
 'droplet scRNA (3&prime; / 5&prime; ~1,300-1,400) and Smart-seq2 (1,557) level - <em>in situ</em>, on FFPE - while FFPE '
 'Visium HD (218/cell) trails (probe capture on degraded FFPE RNA). The fresh-frozen <b>StrataMap</b> (hatched: '
 f'<span class="stat">{sm["median_genes_per_cell"]:,.0f}</span> genes / <span class="stat">{sm["median_transcripts_per_cell"]:,.0f}</span> '
 'UMIs per cell) lands in the same band as Atera and the droplet scRNA references - reaching single-cell-grade depth in tissue too, though '
 'on fresh-frozen so it is not a like-for-like swap for the FFPE assays. So deeply-sequenced dissociated scRNA stays the per-cell '
 'sensitivity leader; the best in-tissue platforms (Atera on FFPE, StrataMap on FF) reach routine-droplet-scRNA depth. The targeted '
 'Xenium / Prime panels are off this axis (gene-capped by design). <span class="note">FFPE vs fresh-frozen marked by hatching; fresh-frozen '
 'Visium HD reaches ~2,728 genes/cell (prep callout above). genes/cell is sequencing-depth-tunable for the scRNA, Visium HD and StrataMap methods.</span></p>')
B.append('</div></section>')

# per-cell transcript distribution per sample (reads inside cells)
RIC=json.load(open(f"{OUT}/reads_in_cells_medians.json"))
B.append('<section class="band"><div class="wrap">')
B.append('<div class="kicker"><span class="n">&#9679;</span><h2>Reads inside cells: the full per-cell distribution</h2></div>')
B.append('<p class="sub">The section above compares <b>median</b> genes per cell; this shows the <b>whole distribution</b> of transcripts '
 'captured inside each segmented cell, per sample - the spread, not just the midpoint. Imaging platforms count qv-filtered molecules '
 '(gene-expression features only); the sequencing platforms count UMIs inside the segmentation contours.</p>')
B.append(f'<figure style="margin:18px 0 0"><img src="{b64("26_reads_in_cells")}" alt="Per-cell transcript distribution per sample" '
 'style="width:100%;border:1px solid var(--line);border-radius:12px;display:block"/></figure>')
B.append('<p class="sub" style="margin-top:14px">The medians (labelled) track the depth story exactly: targeted '
 f'<b>Xenium Prime 5K</b> lowest at <span class="stat">{RIC["Xenium Prime 5K"]:,.0f}</span> and <b>Xenium 313</b> at '
 f'<span class="stat">{RIC["Xenium 313"]:,.0f}</span> transcripts/cell (few, curated genes), the whole-transcriptome platforms far higher - '
 f'<b>Atera</b> <span class="stat">{RIC["Atera (WTx)"]:,.0f}</span>, fresh-frozen <b>StrataMap</b> '
 f'<span class="stat">{RIC["StrataMap (poly-A)"]:,.0f}</span>, and the deeply-sequenced <b>fresh-frozen Visium HD</b> highest at '
 f'<span class="stat">{RIC["Visium HD 6.5mm FF cell"]:,.0f}</span> (vs FFPE Visium HD 11 mm at '
 f'<span class="stat">{RIC["Visium HD 11mm cell"]:,.0f}</span>). The <b>width</b> matters as much as the median: the sequencing platforms '
 'show a long high tail (large cells accumulate many UMIs), while the imaging panels are tighter (per-molecule counting on a fixed panel). '
 '<span class="note">Transcripts/cell on a log10 axis; each violin subsampled to 30k cells. Imaging = GEX features only; StrataMap = 1 µm features '
 'aggregated into the demo segmentation (718k cells). Fresh-frozen samples reach deeper than their FFPE counterparts.</span></p>')
B.append('</div></section>')

# vignette: Atera vs CosMx WTx (cross-vendor whole-transcriptome imaging)
B.append('<section class="band"><div class="wrap">')
B.append('<div class="kicker"><span class="n">vs</span><h2>Whole-transcriptome imaging has a rival: Atera vs CosMx</h2></div>')
B.append('<p class="sub">Atera is not the only instrument that reads the whole transcriptome in situ. Bruker/NanoString&rsquo;s '
 '<b>CosMx WTx</b> images ~18,900 genes optically as well - and the public <b>CosMx Human Multiomic Breast</b> section adds a '
 '<b>64-plex protein</b> panel on the same cells. Running the identical recompute on that breast FFPE section '
 f'(<b>{cos["n_units"]:,}</b> cells, <b>{cos["n_gex_genes"]:,}</b> genes) puts the two whole-transcriptome imagers head to head - '
 'a genuinely matched pair (both optically decode single molecules and segment native cells, so per-cell depth is directly comparable).</p>')
B.append(f'<figure style="margin:18px 0 0"><img src="{b64("13_atera_vs_cosmx")}" alt="Atera vs CosMx WTx head-to-head" '
 'style="width:100%;border:1px solid var(--line);border-radius:12px;display:block"/></figure>')
B.append('<p class="sub" style="margin-top:14px">On matched breast FFPE, <b>Atera leads on RNA</b>: about '
 f'<span class="stat">{cx_depth_g:.1f}&times;</span> the per-cell depth (1,543 vs 977 genes/cell; 2,116 vs 1,319 transcripts), '
 f'~<span class="stat">{cx_pergene:.1f}&times;</span> more signal per gene on the <b>{cx_shared:,}</b> genes both panels share, '
 f'and a dramatically cleaner background - its negative-control probes and unused codewords fire ~<span class="stat">{cx_bg_neg:.0f}&times;</span> '
 f'and ~<span class="stat">{cx_bg_cw:.0f}&times;</span> less often relative to gene signal (10x&rsquo;s error-correcting codebook and '
 f'optical decoding; CosMx decoding-error/falsecode counts run ~{cos["falsecode_frac_of_gex"]*100:.1f}% of gene signal here vs Atera&rsquo;s '
 f'{_ate["negctrl_codeword_frac_of_gex"]*100:.3f}%). The two still agree on the biology, but less than any 10x-to-10x pair in this '
 f'benchmark (Spearman <span class="stat">{cx_conc:.2f}</span> vs 0.59-0.88) - expected across vendors with different probe chemistries.</p>')
B.append('<div class="card" style="border-left:4px solid #A6761D;margin-top:16px">'
 '<h4 style="margin:.1em 0 .4em;font-size:15px">Where CosMx wins: same-section proteogenomics, and availability today</h4>'
 '<p style="color:var(--muted);font-size:14.5px;margin:0">This dataset measured <b>64 proteins on the very same cells</b> - true '
 'RNA + protein multiomics on one section, which Atera (RNA-only) does not offer - and the CosMx SMI is <b>shipping now</b>, whereas '
 'Atera is a preview (ships H2 2026). So the choice is task-shaped: for maximum in-situ RNA sensitivity and specificity, Atera; for '
 'same-section RNA + protein on today&rsquo;s hardware, CosMx WTx.</p></div>')
B.append('<p class="note" style="margin-top:10px">One public breast FFPE section each (CosMx Human Multiomic Breast flatfiles; Atera '
 '10x preview) - directional, not a spec sheet. Both recomputed identically from raw counts; background = per-control-feature count '
 'rate relative to per-gene signal (lower = cleaner). CosMx cells are smaller (median ~86 µm²), which lifts its transcripts/mm² but '
 'not per-cell depth.</p>')
B.append('</div></section>')

# new entrant: Illumina StrataMap (recomputed segmentation-free from the raw 1um-SBC matrix)
B.append('<section class="band"><div class="wrap">')
B.append('<div class="kicker"><span class="n">new</span><h2>A new entrant that breaks the frontier: Illumina StrataMap</h2></div>')
B.append('<p class="sub">Illumina launched <b>StrataMap Spatial</b> (8 June 2026; formerly "Illumina Spatial Solution"), its first '
 'spatial transcriptomics product: <b>sequencing-based</b> like Visium HD but with <b>poly(A) capture</b> instead of a probe panel, so '
 'it is whole-transcriptome in the <em>strict</em> sense. Its public breast-cancer demo (10&nbsp;µm fresh-frozen, NovaSeq X, DRAGEN '
 'Spatial, IDC grade 3) ships a raw <b>1&nbsp;µm spatial-barcode matrix</b> '
 f'({sm["n_gex_genes_annotation"]:,} gene models &times; {sm["n_units"]/1e6:.0f}&nbsp;M features, '
 f'{sm["total_gex_transcripts_all"]/1e9:.1f}&nbsp;billion transcripts). Recomputed the same <b>segmentation-free</b> way as the '
 '8&nbsp;µm-bin check above, it is a full member of every comparable view in this report - the table, the per-gene sensitivity and '
 'breadth-vs-depth figures, and the concordance heatmap - so it is not treated as a footnote.</p>')
B.append('<p class="sub" style="margin-top:12px">And it <b>leads</b>: on the flagship metric - transcripts/mm²/gene on the 304 shared '
 f'genes - StrataMap records <span class="stat">{sm["L1_transcripts_per_mm2_per_gene"]:,.0f}</span>, '
 f'~<span class="stat">{sm_vs_xen:.1f}&times;</span> the targeted 313-plex Xenium ({std["L1_transcripts_per_mm2_per_gene"]:,.0f}) that '
 f'had led every other tier, while covering <b>{sm["n_pc_detected"]:,} protein-coding genes</b> (on par with Atera / Visium HD) '
 f'<b>plus {sm["n_noncoding_detected"]:,} noncoding / pseudogene features</b> those probe sets cannot see. See it top-right of the '
 'breadth-vs-depth figure above.</p>')
B.append('<div class="grid3" style="margin-top:16px">')
B.append('<div class="card" style="border-left:4px solid var(--accent)"><div class="big">750 mm²</div><h4>Capture area &middot; the largest here</h4>'
 f'<p>7.5&nbsp;cm² spec (this demo section covers ~{sm["area_mm2"]:.0f}&nbsp;mm²), vs Atera 500, Xenium 235, Visium HD 42-121.</p></div>')
B.append('<div class="card" style="border-left:4px solid var(--accent2)"><div class="big">1 µm</div><h4>Features &middot; sub-cellular</h4>'
 '<p>A 1&nbsp;µm poly(A)-capture grid on NovaSeq / NextSeq, finer than Visium HD&rsquo;s 2&nbsp;µm. The demo also ships cell + nuclei '
 'segmentation contours.</p></div>')
B.append(f'<div class="card" style="border-left:4px solid #2E6FAF"><div class="big">{sm["n_pc_detected"]/1000:.0f}k + {sm["n_noncoding_detected"]/1000:.0f}k</div>'
 '<h4>Genes &middot; coding + noncoding</h4>'
 '<p>Protein-coding (panel-comparable, in the table) plus tens of thousands of noncoding/pseudogene features - the "~2&times; genes vs probe-based" claim.</p></div>')
B.append('</div>')
B.append('<p class="sub" style="margin-top:16px"><b>Per-cell, and the catch on cell types.</b> Aggregating the 1&nbsp;µm features into the '
 f'demo&rsquo;s segmentation contours (718k cells, 75% of features fall inside a cell) gives a median '
 f'<span class="stat">{sm["median_transcripts_per_cell"]:,.0f}</span> transcripts and <span class="stat">{sm["median_genes_per_cell"]:,.0f}</span> '
 'genes per cell - richer than Atera, as expected for fresh-frozen. But note the mirror image of the frontier: on the <b>304 shared marker '
 f'genes</b> a StrataMap cell holds only <span class="stat">{sm["L1_median_transcripts_per_cell"]:.0f}</span> transcripts vs Xenium&rsquo;s '
 f'{std["L1_median_transcripts_per_cell"]:.0f} and Atera&rsquo;s {wta["L1_median_transcripts_per_cell"]:.0f} - the targeted panels still '
 'concentrate more reads on specific genes per cell, which is why StrataMap is not automatically "best" for rare-cell work. '
 '<b>Cell types are the real gap:</b> the demo ships segmentation geometry but <b>no cell-type labels</b>, and naive marker typing on this '
 'deep whole-transcriptome data gives implausible proportions (~10% mast cells), so StrataMap is deliberately left out of the per-cell-type '
 'figure below - proper annotation needs a reference this public demo does not include.</p>')
RECON=json.load(open(f"{OUT}/reconcile_ref_stats.json"))
B.append('<p class="note" style="margin-top:8px">StrataMap is <b>fresh-frozen only</b> (poly(A) needs intact RNA) - its fair peers are the '
 'fresh-frozen data here, and it does not serve <b>FFPE</b> (the "no" in the application matrix). Gene count is bounded to '
 'protein-coding (the raw 61,906 is full GENCODE). At the FGCZ all-in price (CHF 9,385 / slide) its high yield makes it the cheapest per transcript here; per area it is lowest on paper (usable 13×48=624 mm²) but that is best-case: its SBC registration mandates strict &gt;1 mm inter-sample gaps (imaging platforms have no such limit), so much of the strip is blank and the real per-area cost lands near Atera. One public demo section (IDC grade 3) recomputed.</p>')
B.append('<p class="note" style="margin-top:8px"><b>Reference reconciliation.</b> The public StrataMap demo is aligned to an older GENCODE '
 f'(~v41, GRCh38.p13; {RECON["stratamap_models"]:,} gene models); FGCZ&rsquo;s current reference is <b>{RECON["p14_release"]}</b>. This matters '
 f'only if the gene set moved - it did not: <span class="stat">{RECON["ensg_stable_pct"]}%</span> of StrataMap&rsquo;s genes carry the same Ensembl ID in p14 '
 f'(only {RECON["ensg_retired"]:,} retired), the protein-coding set is <span class="stat">{RECON["pc_present_p14_pct"]}%</span> stable, and all '
 f'<span class="stat">{RECON["core_in_p14"]}/{RECON["core_n"]}</span> genes of the shared marker core are present in <b>both</b> references. '
 'The ~1,400 symbol changes are almost all pseudogenes / novel transcripts, and the HGNC drift on real genes (KARS&rarr;KARS1 etc.) is already '
 'alias-resolved in the merge. So the version gap does not affect any cross-platform number here.</p>')
B.append('</div></section>')

# StrataMap data check (visual sanity check of the raw data behind the numbers)
B.append('<section class="band" style="background:#fff;border-top:1px solid var(--line);border-bottom:1px solid var(--line)"><div class="wrap">')
B.append('<div class="kicker"><span class="n">new</span><h2>StrataMap, seen directly: does the data hold up?</h2></div>')
B.append('<p class="sub">Every StrataMap number above comes out of a heavy pipeline (2.6 billion matrix entries, a 1 µm '
 'grid, a rasterised segmentation). So here is the raw data itself - the check that those numbers describe real tissue and '
 'not a coordinate bug.</p>')
B.append(f'<figure style="margin:18px 0 0"><img src="{b64("16_stratamap_datacheck")}" alt="StrataMap data sanity check" '
 'style="width:100%;border:1px solid var(--line);border-radius:12px;display:block"/></figure>')
B.append('<p class="sub" style="margin-top:14px"><b>A</b> - gridding the 1 µm features to 8 µm bins reproduces a breast tumour '
 'section with real histology: bright tumour nests against darker stroma, sharp tissue edges. The coordinates and the binning are '
 'right. <b>B</b> - the <b>717,578 segmented cells</b> tile that same tissue, confirming the SBC&rarr;cell assignment (75% of 1 µm '
 'features fall inside a cell). <b>C</b> - per-cell counts are cleanly log-normal, no zero-inflated spike or bimodality. '
 '<b>D</b> - genes rise smoothly with transcripts along one tight band: a normal library-complexity curve, so no cell is gaining '
 'genes from ambient bleed. <span class="note">Panels use <b>protein-coding</b> genes (median 2,925 tx / 1,514 genes per cell); the '
 'table quotes all-gene values (3,109 / 1,572). One caveat visible in A and B: the left margin (x&nbsp;&asymp;&nbsp;1-2 mm) is '
 'markedly sparser - a lower-quality tissue edge that a per-cell QC filter would trim.</span></p>')
B.append('<h3 style="margin:26px 0 0;font-size:18px;font-weight:680">And why does its matrix have ~62,000 genes?</h3>')
B.append('<p class="sub" style="margin-top:8px">Because poly(A) sequencing is quantified against the <b>entire Ensembl/GENCODE gene '
 'annotation</b> - there is no probe panel to define the gene space. The 61,906 rows are gene <em>models</em>, and only '
 '<b>32% are protein-coding</b>; the rest are lncRNA (27%), small/misc RNA, pseudogenes and TEC entries, most detected at a '
 'handful of counts. That is why quoting "61,906 genes" against Atera&rsquo;s 18,028-gene panel would be a category error - the '
 'like-for-like number is StrataMap&rsquo;s <b>19,943 protein-coding</b> models (19,025 detected), which is what the table uses.</p>')
B.append(f'<figure style="margin:16px 0 0"><img src="{b64("17_stratamap_biotypes")}" alt="StrataMap gene biotype composition" '
 'style="width:100%;border:1px solid var(--line);border-radius:12px;display:block"/></figure>')
B.append('<p class="note" style="margin-top:10px">*"newer annotation" = 14,970 gene IDs present in StrataMap&rsquo;s reference but not '
 'in our GRCh38.p13 (Ensembl r102) GTF - a newer GENCODE release, overwhelmingly lncRNA/pseudogene/TEC entries. The genuine '
 'capability this buys is real (noncoding RNA the probe panels cannot see at all) - it is the <em>headline gene count</em> that is '
 'not comparable, not the biology.</p>')
B.append('</div></section>')

# ---------------------------------------------------------------------------------------------
# On the horizon: Singular Genomics G4X and Element AVITI24 / Teton. NEITHER is measured - there is
# no public tissue data for either - so this section is explicitly a spec-sheet tier, walled off
# from every recomputed number in the report. Sources and dates are given per claim.
# ---------------------------------------------------------------------------------------------
B.append('<section class="band"><div class="wrap">')
B.append('<div class="kicker"><span class="n">next</span><h2>On the horizon: Singular G4X and '
         'Element AVITI24</h2></div>')
B.append('<div class="card" style="border-left:4px solid #B00;margin-bottom:20px">'
 '<h4 style="margin:.1em 0 .4em;font-size:15px">Read this section differently from the rest of the '
 'report</h4>'
 '<p style="color:var(--muted);font-size:14.5px;margin:0">Every number in the six columns above was '
 '<b>recomputed from a count matrix</b> on this hardware. Nothing below was. Both platforms here are '
 '<b>vendor specifications</b>, because for neither of them can we obtain data: <b>G4X</b> ships no '
 'open dataset (the vendor download is behind a contact form, the AGBT data link no longer resolves, '
 'and the one HuBMAP G4X deposit returns 403 without a token), and <b>AVITI24 / Teton</b> has an open '
 'download but it is a <b>simulated</b> run of cultured cells. To confirm those are real absences and '
 'not failed searches: a GEO platform query returns <b>0</b> records for G4X and <b>0</b> for Teton '
 'against <b>57</b> for "Element AVITI" as a positive control, and G4X appears in <b>none</b> of the '
 'three 2025&ndash;26 systematic FFPE spatial benchmarks. Treat everything below as a claim with a '
 'date on it, not a measurement.</p></div>')
B.append('<div class="grid3">')
B.append('<div class="card" style="border-left:4px solid #8C6BB1">'
 '<h4 style="margin:.1em 0 .45em;font-size:15px">Singular Genomics G4X &middot; a real FFPE '
 'competitor, in the targeted tier</h4>'
 '<p>In-situ sequencing: padlock probes, rolling-circle amplification, sequenced on the instrument. '
 '<b>FFPE</b>, subcellular, and it adds two things Xenium does not bundle - <b>18-plex protein</b> and '
 'fluorescent H&amp;E on the same section. <b>Commercially launched 2026-02-18, purchasable in the '
 'United States</b>; no EU or ex-US availability has been announced. Catalogue panels include a '
 '<b>307-gene breast</b> panel, and the ceiling is <b>500-plex</b>.</p>'
 '<p style="margin-top:.6em"><b>Where it would land in this report:</b> squarely against '
 'Xenium&nbsp;313-plex and Prime&nbsp;5K, <em>not</em> against the whole-transcriptome tier. At '
 '500&nbsp;genes maximum it cannot enter the L2 track at all, and its published per-cell sensitivity '
 'exists only as an unlabelled figure - so the one axis that matters most in this comparison, '
 'transcripts/mm²/gene, is unknown.</p></div>')
B.append('<div class="card" style="border-left:4px solid #7BA05B">'
 '<h4 style="margin:.1em 0 .45em;font-size:15px">Element AVITI24 + Teton &middot; not a tissue '
 'platform yet</h4>'
 '<p>This is the one to be clear about, because the vocabulary overlaps and the category does not. '
 'Teton CytoProfiling is <b>shipping today</b> - 350 RNA targets, up to 138 proteins, 6 morphology '
 'markers, &lt;250&nbsp;nm resolution - but the spec sheet&rsquo;s sample types are '
 '<b>&ldquo;adherent cells, cell suspensions&rdquo;</b>. Cells are seeded into the flow cell. There is '
 'no FFPE, no fresh-frozen, <b>no tissue section</b>.</p>'
 '<p style="margin-top:.6em">Tissue is on the <b>roadmap, not the price list</b>: FFPE and '
 'fresh-frozen via Direct In Sample Sequencing is stated for <b>H2 2026</b>, and protein in tissue for '
 '<b>2027</b>. Until the tissue product ships, AVITI24 is a cell-culture phenotyping instrument and '
 'does not belong in a breast-cancer-section comparison. It is in this report so that the question '
 '&ldquo;why is AVITI24 not in the table?&rdquo; has an answer.</p></div>')
B.append('<div class="card" style="border-left:4px solid var(--accent)">'
 '<h4 style="margin:.1em 0 .45em;font-size:15px">What we would do next</h4>'
 '<p><b>G4X is worth one email.</b> It is FFPE, it has a catalogue breast panel, and its output '
 'includes a <b>per-transcript table</b> carrying x, y, z, a confidence score, an in-nucleus flag and '
 'a cell id - everything the diffusion measurement in section 03b needs. Its segmentation is plain '
 'nuclear expansion, with neither the algorithm nor the expansion distance published, which is exactly '
 'the setup where spillover matters most. <b>Nobody has published a diffusion or spillover measurement '
 'on G4X.</b> Requesting the form-gated breast dataset would let us run every recomputed metric in '
 'this report on it, and produce a number that does not currently exist anywhere.</p>'
 '<p style="margin-top:.6em"><b>AVITI24: revisit when the tissue chemistry ships.</b> Nothing to '
 'benchmark before then.</p></div>')
B.append('</div>')
B.append('<h3 style="margin:28px 0 10px;font-size:18px;font-weight:680">The claimed specifications, '
         'side by side with what we measured</h3>')
_HZ = [
 ("Modality",            "in-situ sequencing (padlock + RCA)", "polony sequencing of barcodes from cells seeded in the flow cell"),
 ("Sample types",        "FFPE (fresh-frozen for Direct-Seq)", "adherent cells and cell suspensions only &mdash; <b>no tissue</b>"),
 ("RNA plex",            "500 max; breast catalogue panel 307; 1,300 showcased (roadmap)", "350 targets"),
 ("Whole transcriptome", "<b>no</b>; Direct-Seq reads up to 100 bases of variable RNA, early access H2 2026", "3&prime; poly-A transcriptome stated for the Atlas/DISS high-output config"),
 ("Protein plex",        "18 (16 + 2 add-on)", "up to 138 (surface, intracellular, phospho)"),
 ("Resolution",          "&ldquo;subcellular&rdquo;; no µm figure published", "&lt; 250 nm"),
 ("Area per run",        "4&ndash;10 cm² per flow cell depending on config, up to 4 flow cells", "10 cm² per flow cell, 2 flow cells per run"),
 ("Segmentation",        "provided: nuclear mask + expanded-nuclear whole-cell approximation; algorithm and expansion distance not published; membrane-based in development", "provided: cell and nuclear masks; model not named"),
 ("List price",          "$240 per sample, $0.0008 per cell; instrument price not published", "$424,000 instrument ($150,000 as an AVITI upgrade); kit prices not published"),
 ("Status, 2026-08",     "<b>GA, United States only</b> since 2026-02-18", "<b>GA for cultured cells</b>; tissue H2 2026, protein in tissue 2027"),
 ("Public tissue data",  "<b>none</b> &mdash; form-gated, dead AGBT link, HuBMAP deposit token-protected", "<b>none</b> &mdash; the only open download is a simulated run"),
 ("Independent evaluation", "one proof-of-concept paper (4 HNSCC FFPE, 350 probes + 15 proteins); no sensitivity, specificity or spillover measured", "<b>none</b> &mdash; both methods preprints are authored by the vendor"),
 ("Diffusion / spillover published", "<b>nothing, by anyone</b>", "<b>nothing, by anyone</b>"),
]
_t = ('<div class="tablewrap"><table class="master"><thead><tr><th></th>'
      '<th class="pcol" style="--pc:#8C6BB1"><span class="pname">Singular G4X</span></th>'
      '<th class="pcol" style="--pc:#7BA05B"><span class="pname">Element AVITI24 / Teton</span></th>'
      '</tr></thead><tbody>')
for lab, a, b_ in _HZ:
    _t += f'<tr><td class="rlab">{lab}</td><td>{a}</td><td>{b_}</td></tr>'
_t += "</tbody></table></div>"
B.append(_t)
B.append('<p class="note" style="margin-top:12px">Two vendor claims deliberately <b>not</b> repeated '
 'above, because they do not survive reading the vendor&rsquo;s own numbers. G4X&rsquo;s launch release '
 'says &ldquo;128 samples and 40&nbsp;cm² per run&rdquo;; the specification page makes that impossible '
 'in one run - 128 samples requires four of the high-count flow cells, which together give about '
 '26&nbsp;cm², while 40&nbsp;cm² requires four of the large-area flow cells, which together take 40 '
 'samples. It is a maximum of each, not a single configuration. And the &ldquo;4 billion cells across '
 '20,000 samples&rdquo; figure is internal validation with no independent confirmation. '
 'Element&rsquo;s headline sensitivity, &ldquo;1 million mean counts per mm²&rdquo;, is footnoted in '
 'the spec sheet as one panel in <b>HeLa cells</b> - a cell line on glass, which is not comparable to '
 'any tissue number in this report.</p>')
B.append('</div></section>')

# section 03: specificity via cross-platform agreement
B.append('<section class="band"><div class="wrap">')
B.append('<div class="kicker"><span class="n">03</span><h2>Specificity: do the platforms agree?</h2></div>')
B.append('<p class="sub" style="margin-bottom:18px">Specificity is measured differently per modality - imaging carries '
 f'negative-control probes (Atera <b>{wta["negctrl_probe_frac_of_gex"]*100:.3f}%</b>, standard panel '
 f'<b>{std["negctrl_probe_frac_of_gex"]*100:.3f}%</b> of signal, in the table); Visium HD has no equivalent. The measure that works '
 'across <em>both</em> modalities is whether the platforms agree on the biology.</p>')
B.append(figblock("f-conc","06_concordance_L1","Platforms agree on the biology",
   f'<p>Pseudobulk expression on the {L1N} shared genes correlates across the whole-transcriptome comparators (Spearman 0.59-0.88; '
   'Prime 5K is scored separately in the L2 track and omitted here so its 117 zero-filled core genes cannot deflate the correlation). '
   '<b>StrataMap included</b> (it agrees with the others at 0.60-0.79, closest to Visium HD 11 mm). '
   'High concordance means the differences between platforms are <em>sensitivity</em>, not artefact - each measures the same '
   'underlying signal, just at different depth - which is what lets the per-gene sensitivity comparison be read as real.</p>'))
B.append('<div class="card" style="border-left:4px solid var(--accent);margin-top:18px">'
 '<h4 style="margin:.1em 0 .4em;font-size:15px">How comparable are the tissue sections themselves?</h4>'
 '<p style="color:var(--muted);font-size:14.5px;margin:0">These are <b>not the same cells</b>. Each platform ran on a '
 '<b>different physical section</b> of breast cancer (serial or near-serial sections from the public 10x / vendor releases; and for '
 'StrataMap and the fresh-frozen Visium HD, a <b>different block entirely</b>). Cell-by-cell agreement is therefore neither expected '
 'nor the aim. What is held roughly constant is the <b>tissue type and biology</b> (breast IDC), and every headline number is an '
 '<b>intensive, section-size-independent</b> metric (per mm², per gene, per cell), so the incidental area or cellularity of a given '
 'section cannot drive the ranking. The concordance above (Spearman <b>0.59&ndash;0.88</b> on the shared genes) confirms the platforms '
 'measure the <b>same broad signal</b> with no gross artifact - but read it as a <b>floor check, not proof of equivalence</b>. '
 'Pseudobulk rank-correlation is dominated by the conserved expression-<em>abundance</em> structure (a few genes high everywhere, most '
 'low everywhere), so it is a lenient test and is confounded with platform chemistry. A direct control makes this concrete: a '
 '<b>different tissue</b> - our cervix sections run on the same Atera platform - still correlates with the breast at Spearman '
 '<b>0.67</b> on the 304 markers and <b>0.83</b> on the broader 4,843-gene set, i.e. <em>as high as</em> the same-tissue '
 'cross-platform values. In other words the correlation cannot, on its own, certify that two sections are the same tissue. Caveats '
 'that genuinely differ between these sections: FFPE block age and fixation, tumour content, and stroma/immune fraction; and the '
 'fresh-frozen datasets are a different block and prep - which is exactly why their per-cell advantages are flagged as '
 '<b>prep, not platform</b>. Net: the design is comparable enough for <b>per-gene sensitivity and cost-per-area</b> conclusions '
 '(the intensive metrics, not the correlation, carry that), but it is <b>not</b> a substitute for running two platforms on '
 '<b>adjacent sections of one block</b> if cell-level concordance is the goal.</p></div>')
B.append('</div></section>')

# ---------------------------------------------------------------------------------------------
# section 03b: diffusion / lateral spillover. Nothing in sections 02-03 can catch a platform that
# is sensitive, agrees on pseudobulk, and still puts the signal in the wrong PLACE.
# ---------------------------------------------------------------------------------------------
def _dif(k, src, f):
    return (DIF.get(k, {}) or {}).get(src, {}).get(f)

def _um(v, nd=0, dash="n.r."):
    return dash if v is None else f"{v:.{nd}f}&nbsp;µm"

def difftable():
    cols = [k for k in PLAT if k in DIF]
    if not cols:
        return '<p class="note">Diffusion metrics not computed yet (run 29_diffusion.py).</p>'
    head = "".join(f'<th class="pcol" style="--pc:{CLR[k]}"><span class="pname">'
                   f'{NAME[k].replace(" 6.5 mm"," 6.5").replace(" 11 mm"," 11")}</span></th>'
                   for k in cols)
    def r(label, fn, note=None):
        n = (f'<span class="rnote" title="{note}">?</span>' if note else "")
        return (f'<tr><td class="rlab">{label}{n}</td>'
                + "".join(f"<td>{fn(k)}</td>" for k in cols) + "</tr>")
    t = ('<div class="tablewrap"><table class="master"><thead><tr><th></th>' + head
         + "</tr></thead><tbody>")

    def lam_cell(k, s):
        v = _dif(k, s, "lambda_um")
        if v is not None:
            return f"{v:.0f}&nbsp;µm"
        why = _dif(k, s, "lambda_withheld_because") or "not measurable"
        return f'<span class="rnote" title="withheld: {why}" style="color:#999">withheld</span>'

    t += grp("Signal outside any cell &mdash; no domain, no fit, nothing to tune")
    t += r("Transcripts not assigned to any cell",
           lambda k: (lambda v: '<span style="color:#999">n/a</span>' if v is None
                      else f"<b>{v*100:.1f}%</b>")(
               (DIF.get(k, {}) or {}).get("off_cell_frac")),
           note=("imaging platforms: decoded molecules the pipeline could not place in a cell. "
                 "StrataMap: 1 um features falling outside every segmentation contour. Visium HD "
                 "8 um bins have no equivalent - a bin is not a cell, so there is nothing to be "
                 "outside of"))
    t += grp("Is the leak one-sided? (the Visium v1 signature)")
    for src, nice in (("immune", "Immune"), ("epithelial", "Epithelial"), ("stromal", "Stromal")):
        t += r(f"Directional drift, {nice} source",
               lambda k, s=src: (lambda v: '<span style="color:#999">n.r.</span>' if v is None else
                                 (f'<b style="color:#B00">{v:.1f}&nbsp;µm</b>' if v >= DIF_BIN_UM
                                  else f'<b style="color:#1a7f5a">&lt;&nbsp;{DIF_BIN_UM:.0f}&nbsp;µm</b>'))(
                   _dif(k, s, "drift_equiv_um")),
               note=("the one-sided offset in um that would reproduce the measured directional "
                     "bias, calibrated against injected offsets. Floor = one 8 um bin. This row "
                     "needs no curve fit, so it is not gated like lambda below"))
    t += grp("How far the signal reaches (spillover decay length &lambda;)")
    for src, nice in (("immune", "Immune"), ("epithelial", "Epithelial"), ("stromal", "Stromal")):
        t += r(f"&lambda;, {nice} source",
               lambda k, s=src: lam_cell(k, s),
               note=("distance over which the compartment's own marker signal decays outside its "
                     "own domain, on the shared 8 um lattice. Withheld where the fit did not "
                     "converge, where the in-domain / far-field contrast was under 1.5x, or where "
                     "rolling the domain to the wrong part of the section reproduced the same "
                     "decay - hover the cell for which one"))
    t += r("&lambda; stable when the marker halves are swapped?",
           lambda k: (lambda v: '<span style="color:#999">n.r.</span>' if v is None else
                      (f'<b style="color:#1a7f5a">yes</b> ({v:.2f}&times;)' if 0.7 <= v <= 1.4
                       else f'<b style="color:#B00">no</b> ({v:.2f}&times;)'))(
               _dif(k, "immune", "swap_lambda_ratio")),
           note=("immune source, the only compartment measurable on all six platforms. The marker "
                 "set is split in two: one half defines the domain, the other is measured. "
                 "Exchanging the halves must not change lambda if lambda is a property of the "
                 "platform rather than of those particular genes"))
    t += r("Domain contrast (in-domain / far-field)",
           lambda k: (lambda v: '<span style="color:#999">n.r.</span>' if v is None
                      else f"{v:.1f}&times;")(_dif(k, "immune", "contrast")),
           note=("immune source; higher means the signature is better confined to its own domain. "
                 "This is the quantity the 1.5x gate is applied to"))
    t += r("Leak elongated along one axis? (&lambda;max/&lambda;min)",
           lambda k: (lambda v: '<span style="color:#999">n.r.</span>' if v is None
                      else f"{v:.1f}&times;")(_dif(k, "immune", "aniso_ratio")),
           note=("immune source. Elongated but NOT one-sided means tissue architecture (ducts, "
                 "nests, vessels) rather than transport; the isotropic positive control returns "
                 "1.08x"))
    return t + "</tbody></table></div>"

_pc = DST.get("_all_passed")
_ctl = [c.get("passed") for v in (DREG or {}).values() if isinstance(v, dict)
        for c in (v.get("controls") or {}).values()]
_cal = DCAL.get("drift_mag") or []
_d8 = f"{_cal[2]:.2f}" if len(_cal) > 2 else "?"

B.append('<section class="band" style="background:#fff;border-top:1px solid var(--line);'
         'border-bottom:1px solid var(--line)"><div class="wrap">')
B.append('<div class="kicker"><span class="n">03b</span><h2>Diffusion: does the signal stay where '
         'it was made?</h2></div>')
B.append('<p class="sub" style="margin-bottom:6px">Nothing measured so far would catch a platform '
 'that is sensitive, agrees with the others on pseudobulk, and still reports the signal in the '
 '<em>wrong place</em>. That was <b>Visium v1</b>&rsquo;s defining weakness: mRNA moved laterally '
 'under the section during permeabilisation, so a transcript was recovered a systematic distance '
 'away from the cell that made it. A platform can score well on every row above and still have it. '
 'So it is measured directly, and it is measured <b>two independent ways</b>, because "drift" can '
 'mean two different failures.</p>')
B.append('<div class="grid3" style="margin-top:16px">')
B.append('<div class="card" style="border-left:4px solid var(--accent)">'
 '<h4 style="margin:.1em 0 .45em;font-size:15px">1. Is the leak one-sided?</h4>'
 '<p>Each compartment&rsquo;s markers are split into two halves. One half defines the compartment&rsquo;s '
 'domain; the other is measured <em>outside</em> it, ring by ring, as a function of distance. That '
 'gives a decay length &lambda;, and - the part that answers the drift question - whether the excess '
 'sits <b>evenly all round</b> the domain (ordinary diffusion or blur) or <b>preferentially on one '
 'side</b> (transport). All six platforms, one definition, on the same 8&nbsp;µm lattice used for the '
 'segmentation-free sensitivity line, so no cell-vs-bin confound enters.</p></div>')
B.append('<div class="card" style="border-left:4px solid var(--accent2)">'
 '<h4 style="margin:.1em 0 .45em;font-size:15px">2. Is the capture grid offset from the tissue?</h4>'
 '<p>Test 1 compares two gene sets <em>within</em> one section, so it is blind by construction to a '
 '<b>rigid shift of the whole grid</b> - that moves every gene together. But a rigid shift is exactly '
 'what Visium v1 did. Only the <b>sequencing</b> platforms can fail this way: on Xenium and Atera a '
 'transcript&rsquo;s x,y <em>is</em> an image coordinate from the same optical pass, so there is no '
 'separate grid to mis-register. For Visium HD the barcode lattice and the tissue image are aligned '
 'in software, so the two are cross-correlated directly and the best-fit offset is reported in µm.</p></div>')
B.append('<div class="card" style="border-left:4px solid #B00">'
 '<h4 style="margin:.1em 0 .45em;font-size:15px">Both tests are calibrated first &mdash; on '
 'synthetic data <em>and</em> on Visium v1</h4>'
 f'<p>A "no drift found" result is worth nothing unless the test can find drift that is there. So a '
 f'known offset is <b>injected</b> and has to come back. Test 1: an 8&nbsp;µm one-sided offset '
 f'registers as |D|&nbsp;=&nbsp;{_d8} and a purely isotropic blur calibrates below one bin; a blur '
 f'narrower than one bin is reported as <b>unresolved, not as zero</b>. Test 2: three injected '
 f'offsets were recovered '
 f'{("<b>" + str(sum(1 for c in _ctl if c)) + "/" + str(len(_ctl)) + "</b>") if _ctl else "(pending)"} '
 f'to within one bin. Estimator self-check: '
 f'<b>{"all checks passed" if _pc else "see _selftest.json"}</b>.</p></div>')
B.append('</div>')
B.append(f'<h3 style="margin:28px 0 10px;font-size:18px;font-weight:680">What the two tests found</h3>')
B.append(difftable())

# --- the headline read, computed rather than asserted
_dr = [(_dif(k, s, "drift_equiv_um"), k, s) for k in PLAT if k in DIF
       for s in ("epithelial", "immune", "stromal")]
_dr = [(v, k, s) for v, k, s in _dr if v is not None]
_worst = max(_dr)[0] if _dr else None
_regs = {k: v for k, v in (DREG or {}).items()
         if isinstance(v, dict) and v.get("offset_mag_um") is not None and v.get("valid")}
_regtxt = ", ".join(f'{NAME.get(k,k)} <b>{v["offset_mag_um"]:.1f}&nbsp;µm</b>'
                    for k, v in _regs.items()) or "not yet computed"
B.append('<p class="sub" style="margin-top:18px"><b>The answer is no, on both tests, for every '
 'platform here.</b> ' +
 (f'The largest one-sided drift anywhere in the matrix - six platforms &times; three compartments - '
  f'is <span class="stat">{_worst:.1f}&nbsp;µm</span>, against a detection floor of '
  f'{DIF_BIN_UM:.0f}&nbsp;µm and a test that recovers an injected 8&nbsp;µm offset. ' if _worst is not None else '') +
 f'The grid-to-tissue offset on the two Visium HD sections is {_regtxt} - well under half a bin, and '
 'shifting the grid does not improve its agreement with the tissue image at all, while three injected '
 'offsets per section were recovered exactly. <b>Whatever Visium v1 did, none of these six '
 'platforms is doing it.</b> That is the single most useful thing in this section, and it is a '
 'negative result that only counts because the test was calibrated first.</p>')
_imm = [(_dif(k, "immune", "lambda_um"), k) for k in PLAT if k in DIF]
_imm = sorted([(v, k) for v, k in _imm if v is not None])
_offc = [((DIF.get(k, {}) or {}).get("off_cell_frac"), k) for k in PLAT if k in DIF]
_offc = sorted([(v, k) for v, k in _offc if v is not None])
B.append('<p class="sub" style="margin-top:14px"><b>What they <em>do</em> show is isotropic spillover '
 'over tens of microns</b>, on every platform including the imaging ones. ' +
 (f'On the immune compartment - the only one of the three that survives the quality gates on all '
  f'{len(_imm)} platforms - the decay length runs from '
  f'<span class="stat">{_imm[0][0]:.0f}&nbsp;µm</span> ({NAME[_imm[0][1]]}) to '
  f'<span class="stat">{_imm[-1][0]:.0f}&nbsp;µm</span> ({NAME[_imm[-1][1]]}). ' if len(_imm) >= 2 else '') +
 'The striking thing is how <b>narrow</b> that spread is: imaging and sequencing land in the same '
 'band, and the ranking does not follow modality, plex, or price. Nobody here confines a '
 'compartment&rsquo;s signature to its own boundary, and on every platform the leak reaches '
 '<b>further than one cell diameter</b>. For anyone doing boundary biology - tumour-immune '
 'interface, invasive front, niche definition - that is the number that matters, and it is not a '
 'reason to pick one of these platforms over another.</p>')
if len(_offc) >= 2:
    B.append('<p class="sub" style="margin-top:14px"><b>Where the platforms <em>do</em> separate is '
     'how much signal never lands in a cell at all</b> - and this is the cleanest number in the '
     'section, because it needs no domain, no threshold and no curve fit. ' +
     ", ".join(f'{NAME[k]} <span class="stat">{v*100:.1f}%</span>' for v, k in _offc) +
     '. It rises monotonically with how much of the transcriptome the assay tries to read: the more '
     'genes in play, the larger the share of detected signal the pipeline cannot assign to any cell. '
     'That is a real cost of breadth, it is paid in exactly the place segmentation-dependent '
     'analyses are most fragile, and it is invisible in every sensitivity metric in section 02. '
     '<span class="note">Visium HD is absent from this row by definition, not by omission: its unit '
     'is an 8&nbsp;µm bin, and a bin is not a cell, so there is nothing for signal to be outside '
     'of. Read the imaging figures as pipeline cell-assignment rates and StrataMap&rsquo;s as the '
     'share of 1&nbsp;µm features outside every segmentation contour - closely related, but not '
     'produced by identical software.</span></p>')
# swap-stability roll-up for the limits card: which platforms keep their lambda when the marker
# halves are exchanged. 0.7-1.4x counts as stable (the synthetic control sits at 1.0).
_sw = [(_dif(k, "immune", "swap_lambda_ratio"), k) for k in PLAT if k in DIF]
_sw = [(v, k) for v, k in _sw if v is not None]
_sw_ok  = sorted([r for r in _sw if 0.7 <= r[0] <= 1.4])
_sw_bad = sorted([r for r in _sw if not (0.7 <= r[0] <= 1.4)])
_SWAP_TEXT = (
    f'<b>2. &lambda; is not always a platform constant.</b> Swapping which marker half defines the '
    f'domain leaves it intact on <b>{len(_sw_ok)} of {len(_sw)}</b> platforms for the immune '
    f'compartment ('
    + ", ".join(f"{NAME[k]} {v:.2f}&times;" for v, k in _sw_ok[:3])
    + (", &hellip;" if len(_sw_ok) > 3 else "") + ')'
    + ((' but not on ' + ", ".join(f"<b>{NAME[k]}</b> ({v:.2f}&times;)" for v, k in _sw_bad) + '. ')
       if _sw_bad else '. ')
    + 'On the epithelial compartment <b>both Visium HD sections</b> fail the same check '
      '(0.34&ndash;0.48&times;) while both Xenium tiers pass, so on that chemistry different probes '
      'for the same compartment do not behave alike. Where the swap fails, &lambda; describes those '
      'genes rather than the platform - which is why no "platform X leaks N&times; further than '
      'platform Y" claim appears anywhere in this section. ')

B.append('<div class="card" style="border-left:4px solid #B00;margin-top:18px">'
 '<h4 style="margin:.1em 0 .4em;font-size:15px">Three limits, and why &lambda; is not quoted as a '
 'platform constant</h4>'
 '<p style="color:var(--muted);font-size:14.5px;margin:0">'
 '<b>1. &lambda; is not purely technical.</b> Single epithelial cells genuinely sit in stroma, and '
 'immune cells genuinely infiltrate tumour nests. The measurement cannot separate that real biology '
 'from chemical or optical spillover, so &lambda; is an <b>upper bound</b> on the technical component, '
 'not an estimate of it. The <em>directional</em> test does not share this problem, which is why the '
 'drift result is the one stated as a finding. '
 + _SWAP_TEXT +
 '<b>3. The lattice sets a floor.</b> An 8&nbsp;µm grid cannot resolve a decay shorter than 8&nbsp;µm, '
 'and cannot detect a drift smaller than that either. Everything below the floor is reported as '
 'unresolved rather than as zero. Finer bins exist for Visium HD (2&nbsp;µm) and the imaging platforms '
 'have molecule coordinates, so this floor is a deliberate cost of using <b>one identical unit for all '
 'six platforms</b> - the same trade the 8&nbsp;µm sensitivity check makes. '
 'And as everywhere in this report, these are <b>different sections of different blocks</b>, so a '
 'compartment&rsquo;s architecture is not held constant across columns.</p></div>')
# --- the real-data control: does the machinery find diffusion on a platform that HAS it?
if V1 and V1H and _mean(V1, "offtissue_level_at_first_ring") and _mean(V1H, "offtissue_level_at_first_ring"):
    _v1r = _mean(V1, "offtissue_level_at_first_ring"); _hdr = _mean(V1H, "offtissue_level_at_first_ring")
    _v1d = _mean(V1, "drift_mag");                     _hdd = _mean(V1H, "drift_mag")
    # precomputed so no f-string has to nest its own quote character (py<3.12 parse error)
    _pct = lambda d: ", ".join("%.1f%%" % (v["offtissue_level_at_first_ring"] * 100)
                               for v in d.values())
    _dmg = lambda d: " / ".join("%.3f" % v["drift_mag"] for v in d.values())
    _v1_pct, _hd_pct = _pct(V1), _pct(V1H)
    _v1_dmg, _hd_dmg = _dmg(V1), _dmg(V1H)
    B.append('<h3 style="margin:30px 0 10px;font-size:18px;font-weight:680">The control that '
             'licenses all of the above: Visium v1</h3>')
    B.append('<p class="sub">Injected offsets prove the tests are not blind to an artifact of a '
     '<em>known size</em>. They do not prove the tests are sensitive to diffusion as it actually '
     'happens in tissue. For that you need a platform that demonstrably has the problem, so the '
     'same code was pointed at <b>Visium v1</b> - the chemistry whose lateral diffusion started '
     'this question - using the one handle a Visium capture area gives you for free: <b>it has '
     'spots beyond the tissue</b>. Space Ranger flags them. Any signal recovered from a spot that '
     'contains no tissue got there by moving.</p>')
    B.append('<div class="grid3" style="margin-top:16px">')
    B.append(f'<div class="card" style="border-left:4px solid #B2182B">'
     f'<div class="big">{_v1r*100:.0f}%</div><h4>Visium v1 &middot; signal in the ring just '
     f'<em>outside</em> the tissue</h4><p>As a share of the in-tissue level, averaged over two '
     f'independent sections of block A ({_v1_pct}). '
     f'Nearly half the signal level, in spots holding no tissue at all.</p></div>')
    B.append(f'<div class="card" style="border-left:4px solid #66a61e">'
     f'<div class="big">{_hdr*100:.0f}%</div><h4>Visium HD &middot; the same measurement, the same '
     f'code</h4><p>{_hd_pct} '
     f'for the 6.5 mm and 11 mm sections. <b>{_v1r/_hdr:.1f}&times; less</b> than v1. Same vendor '
     f'lineage, same <code>in_tissue</code> flag, same rings in µm - only the chemistry differs.</p></div>')
    B.append(f'<div class="card" style="border-left:4px solid var(--accent)">'
     f'<div class="big">{_v1d/_hdd:.0f}&times;</div><h4>and v1&rsquo;s leak is that much more '
     f'<em>one-sided</em></h4><p>|D| = {_v1_dmg} on v1 '
     f'against {_hd_dmg} on Visium HD, each judged '
     f'against its own 200-shuffle null. This is the <b>systematic</b> part of the drift, and it is '
     f'the half of the control that a spot-footprint artefact cannot fake.</p></div>')
    B.append('</div>')
    B.append('<p class="sub" style="margin-top:16px"><b>So the tests work, and the negative result '
     'above is a real negative.</b> Pointed at v1 they return a large, distance-dependent, '
     'directional leak in both sections independently. Pointed at the six modern platforms they '
     'return nothing above the detection floor. That is the difference between "we looked and found '
     'nothing" and "we have no idea", and it is the reason this section is worth reading at all.</p>')
    B.append('<div class="card" style="border-left:4px solid #B00;margin-top:16px">'
     '<h4 style="margin:.1em 0 .4em;font-size:15px">Two things this control does <em>not</em> '
     'settle, and one bug it caught</h4>'
     '<p style="color:var(--muted);font-size:14.5px;margin:0">'
     '<b>A v1 spot is 55 µm across on a 100 µm pitch</b>, so a spot 50 µm beyond the detected '
     'boundary can physically overlap tissue the mask missed. That inflates v1&rsquo;s first-ring '
     'number by an unknown amount. It is also <b>isotropic</b>, so it cannot produce the '
     'directional asymmetry - which is exactly why the verdict leans on |D| and not on the leak '
     'size alone. <b>Visium HD is not at zero either</b> (11&ndash;15% in the first ring, and its '
     '11 mm section&rsquo;s |D| does clear its own null): some of that is the same mask-boundary '
     'effect, some is real spillover, and this test cannot separate them. Read the v1-vs-HD '
     '<em>ratio</em>, not HD&rsquo;s absolute value. '
     '<b>And the control earned its keep by breaking the first two versions of its own statistics.</b> '
     'A single shuffle "resolved" a section whose counts had been randomised - one permutation '
     'cannot separate noise from noise. And the fitted decay amplitude turned out to be '
     'unusable as a test statistic: it trades off against the ambient floor once the decay length '
     'is large, so shuffled data fitted <em>larger</em> amplitudes (99th percentile 476%) than the '
     'real profile (54%). Both are recorded in the script header so they do not get reintroduced. '
     'What is reported instead is a model-free rank correlation of distance against counts, and '
     'effect sizes rather than p-values - with 1.5 M off-tissue bins on the 11 mm section, every '
     'p-value is small whatever the truth.</p></div>')
    # --- the same axis extended to all six platforms
    _oth = {k: v for k, v in (DOTA or {}).items()
            if k in PLAT and isinstance(v, dict) and "error" not in v}
    _brg = {k: v for k, v in ((DOTA or {}).get("_vendor_bridge") or {}).items() if v}
    if _oth:
        _fr = sorted((v["offtissue_level_at_first_ring"], k) for k, v in _oth.items())
        _dm = sorted((v["drift_mag"], k) for k, v in _oth.items())
        _v1fr = _mean(V1, "offtissue_level_at_first_ring")
        _v1dm = _mean(V1, "drift_mag")
        _sens = (DOTA or {}).get("_threshold_sensitivity", {})
        B.append('<h3 style="margin:30px 0 10px;font-size:18px;font-weight:680">And the other four '
                 'platforms on that same axis</h3>')
        B.append('<p class="sub">The comparison above only reached Visium HD because it keyed off '
         'the vendor <code>in_tissue</code> flag, and only a barcoded capture array has one. The '
         'imaging platforms and StrataMap have a tissue boundary all the same, and the 8&nbsp;µm '
         'count fields already built for the &lambda; measurement are enough to find it - so the '
         'mask is <b>derived from the data</b>, with one definition for all six, and the leak '
         'outside it measured exactly as for v1. Three things had to be settled before that number '
         'means anything.</p>')
        B.append('<div class="grid3" style="margin-top:16px">')
        B.append('<div class="card"><h4 style="margin:.1em 0 .45em;font-size:15px">A lumen is not '
         'a tissue margin</h4><p>A lumen or a fat globule inside the section has tissue on every '
         'side, so signal reaching it arrives from all directions - nothing like a spot out on the '
         'free margin beyond the section edge, which is what v1&rsquo;s off-tissue spots are. So '
         'the mask has its <b>holes filled</b>, which counts a lumen as tissue and keeps it out of '
         'the margin measurement entirely. Only the <b>exterior</b> component of the complement is '
         'measured - and after hole-filling that is the only component there is.</p></div>')
        B.append('<div class="card"><h4 style="margin:.1em 0 .45em;font-size:15px">A mask built '
         'from the counts is circular</h4><p>Call tissue &ldquo;where counts are high&rdquo;, then '
         'measure counts outside it, and the threshold quietly sets the answer. So everything runs '
         'at <b>three thresholds</b> and the spread appears as a range on every bar below. It '
         'earns its place: one platform&rsquo;s leak level moves <b>19&times;</b> across the three '
         'masks and is simply not usable, while its <em>directionality</em> barely moves. The two '
         'have to be judged separately.</p></div>')
        _bt = ""
        if _brg:
            _bt = " ".join(
                f'{NAME.get(k,k)}: vendor flag <b>{v["vendor_first_ring"]*100:.1f}%</b> vs '
                f'data-derived <b>{v["derived_first_ring"]*100:.1f}%</b> '
                f'({v["ratio"]:.2f}&times;).' for k, v in _brg.items())
        B.append('<div class="card" style="border-left:4px solid var(--accent)">'
         '<h4 style="margin:.1em 0 .45em;font-size:15px">The bridge that ties the two scales '
         'together</h4><p>Visium HD has <em>both</em> a vendor flag and a data-derived mask, so it '
         'is measured both ways on the same section. ' + (_bt or "Not yet computed.") +
         ' Close enough that the derived numbers can be read on the same axis as v1&rsquo;s - which '
         'is the only reason the four platforms without a flag appear here at all.</p></div>')
        B.append('</div>')
        def _lev(key):
            d = (_sens.get(key) or {}).get("level") or {}
            g = [v for v in d.values() if v]
            return (min(g), max(g), (max(g) / min(g)) if g and min(g) > 0 else None) if len(g) > 1 else None
        def _drf(key):
            d = (_sens.get(key) or {}).get("drift") or {}
            g = [v for v in d.values() if v]
            return (min(g), max(g), (max(g) / min(g)) if g and min(g) > 0 else None) if len(g) > 1 else None

        _dsw = {k: _drf(k) for k in _oth}
        _dsw_max = max((v[2] for v in _dsw.values() if v and v[2]), default=None)
        B.append('<p class="sub" style="margin-top:18px"><b>Result: Visium v1 leaks more than any '
         'current platform</b> - in the first 100&nbsp;µm beyond the tissue edge it sits at '
         f'<span class="stat">{_v1fr*100:.0f}%</span> of its in-tissue level, against '
         f'<span class="stat">{_fr[0][0]*100:.0f}&ndash;{_fr[-1][0]*100:.0f}%</span> across the '
         'six. That is the comparison the bridge licenses, and it holds.</p>')
        B.append('<p class="sub" style="margin-top:14px"><b>The <em>directional</em> comparison, '
         'though, does not survive being extended past the vendor flag - so it is not made here.</b> '
         'Running the mask at three thresholds moves the one-sidedness of the leak by '
         + (f'up to <span class="stat">{_dsw_max:.1f}&times;</span> ' if _dsw_max else '')
         + 'on the platforms that have no <code>in_tissue</code> flag: StrataMap alone spans '
         + (f'{_dsw["stratamap_breast"][0]:.3f}&ndash;{_dsw["stratamap_breast"][1]:.3f} '
            if _dsw.get("stratamap_breast") else '')
         + 'across the three masks, which straddles Visium v1&rsquo;s own value - so it can be '
         'made to look more directional than v1 or several times less, depending on where the '
         'boundary is drawn. Two platforms are stable ('
         + ", ".join(f"{NAME[k]} {v[2]:.2f}&times;" for k, v in sorted(
             ((k, v) for k, v in _dsw.items() if v and v[2]), key=lambda kv: kv[1][2])[:2])
         + '), the rest are not. <b>The honest conclusion is that the leak <em>level</em> '
         'transfers to a data-derived mask and the <em>direction</em> does not</b>, so the '
         'directional finding stays where a vendor tissue boundary exists to anchor it: Visium v1 '
         f'at {_v1dm:.3f} against Visium HD at '
         + " / ".join(f"{V1H[k]['drift_mag']:.3f}" for k in V1H) + ', measured from the flag.</p>')

        _xen = _oth.get("stdxenium_breast", {})
        if _xen:
            _xs = _lev("stdxenium_breast")
            B.append('<div class="card" style="border-left:4px solid #B00;margin-top:16px">'
             '<h4 style="margin:.1em 0 .4em;font-size:15px">Two of those six leak levels should not '
             'be read as leaks</h4>'
             '<p style="color:var(--muted);font-size:14.5px;margin:0">'
             f'<b>Xenium 313-plex reads {_xen["offtissue_level_at_first_ring"]*100:.0f}%</b>, well '
             'above the rest, and that is about what the instrument images rather than about the '
             'chemistry. An imaging run covers a region the operator draws around the tissue: on '
             'this section <b>97.5% of the field has transcripts in it</b>, so there is almost no '
             'blank slide to measure and the "exterior" the mask carves out is largely the '
             'low-count <em>rim of the tissue itself</em>. Read it as a ceiling set by geometry. '
             + (f'<b>StrataMap reads {_oth["stratamap_breast"]["offtissue_level_at_first_ring"]*100:.1f}% '
                f'but moves over {_lev("stratamap_breast")[0]*100:.1f}&ndash;'
                f'{_lev("stratamap_breast")[1]*100:.1f}% across the three mask thresholds - a '
                f'{_lev("stratamap_breast")[2]:.0f}&times; swing</b>, so its <em>level</em> is not '
                'a measurement at all and no number for it is quoted in the text above. Its '
                'directionality is a different matter: that moves only '
                + (f'{_drf("stratamap_breast")[2]:.2f}&times; ' if _drf("stratamap_breast") else '')
                + 'over the same thresholds. ' if "stratamap_breast" in _oth
                  and _lev("stratamap_breast") else '')
             + 'This is the general limit of pushing an off-tissue test onto a platform that never '
             'images blank slide, and it is why the directional statistic carries the comparison '
             'while the leak level only supports it.</p></div>')

        _sm = _oth.get("stratamap_breast", {})
        if _sm and _dsw.get("stratamap_breast"):
            _lo, _hi, _sw = _dsw["stratamap_breast"]
            B.append('<div class="card" style="border-left:4px solid #2E6FAF;margin-top:16px">'
             '<h4 style="margin:.1em 0 .4em;font-size:15px">Worth a second look, on a second '
             'section: StrataMap</h4>'
             '<p style="color:var(--muted);font-size:14.5px;margin:0">'
             'At the middle mask threshold StrataMap&rsquo;s off-tissue leak is the most directional '
             f'thing measured anywhere in this report - |D| = <b>{_sm["drift_mag"]:.3f}</b> against '
             f'a shuffled-null 99th percentile of {_sm.get("drift_null_p99", 0):.3f} for that same '
             'section - and its off-tissue signal decays the most steeply with distance of anything '
             f'here (&rho; = {_sm.get("spearman_dist_vs_counts", 0):+.3f}), so it is signal leaving '
             'the tissue rather than a flat background. '
             f'<b>But it is not stable: across the three masks |D| runs {_lo:.3f}&ndash;{_hi:.3f}, '
             f'a {_sw:.1f}&times; swing that straddles Visium v1&rsquo;s own value.</b> So it '
             'cannot be ranked against v1 either way, and it is not counted as a finding. '
             'It is flagged because the mechanism would fit if it were real: of everything in this '
             'comparison StrataMap is the closest relative of Visium v1 - poly(A) capture onto a '
             'barcoded surface, fresh-frozen, with the transcript having to travel to its barcode, '
             'where an imaging platform decodes a molecule wherever it happens to sit. Competing '
             'explanations that this test cannot exclude: a reagent or fluidics gradient across '
             'the flow cell (the per-ring centring removes where the tissue sits, not a genuine '
             'gradient), and the fact that this is <em>one public demo section</em>. '
             'The cheap next step is the same measurement on a second StrataMap section with a '
             'known tissue boundary, which would settle it.</p></div>')
        if os.path.exists(f"{FIG}/fig_offtissue_all.png"):
            B.append(f'<figure style="margin:20px 0 0"><img src="{b64("fig_offtissue_all")}" '
                     'alt="All platforms on the Visium v1 off-tissue axis" style="width:100%;'
                     'border:1px solid var(--line);border-radius:12px;display:block"/></figure>')
            B.append('<p class="note" style="margin-top:10px">Left: counts recovered beyond the '
             'tissue edge against distance, log scale - the two Visium v1 sections solid, the six '
             'current platforms dashed. Middle: the level in the first 100&nbsp;µm, with the black '
             'bar showing the range over three mask thresholds (v1 uses the vendor flag, so it has '
             'no such range). Right: how one-sided each leak is, against the 99th percentile of '
             'that section&rsquo;s own shuffled null.</p>')

    if os.path.exists(f"{FIG}/fig_v1_control.png"):
        B.append(f'<figure style="margin:20px 0 0"><img src="{b64("fig_v1_control")}" '
                 'alt="Visium v1 real-data positive control" style="width:100%;border:1px solid '
                 'var(--line);border-radius:12px;display:block"/></figure>')
        B.append('<p class="note" style="margin-top:10px">Left: counts recovered from spots and '
         'bins holding no tissue, against distance from the nearest tissue-containing spot, log '
         'scale. The two Visium v1 sections (solid) sit far above both Visium HD sections (dashed) '
         'at every distance. Middle: the level in the first ring outside the tissue. Right: how '
         'one-sided each leak is, with the tick marking the 99th percentile of that section&rsquo;s '
         'own shuffled null - v1 clears its null in both sections, Visium HD 6.5 mm does not clear '
         'its own at all.</p>')

if os.path.exists(f"{FIG}/fig_diffusion_lambda.png"):
    B.append(figblock("f-dif1","fig_diffusion_lambda","The decay, measured",
       '<p>Left: the spillover decay length per platform and compartment. The black dash is the same '
       'measurement with the two marker halves <b>swapped</b> - close to the bar means &lambda; is a '
       'property of the platform, far from it means a property of those genes. The red cross is the '
       '<b>displaced-domain null</b>: the value the estimator returns when the domain is rolled to the '
       'wrong part of the section. Where the cross sits at or above the bar, that measurement is '
       'not interpretable, which is what happens to the stromal compartment on several platforms.<br>'
       'Right: the actual radial profiles for the epithelial source, log scale, with the fitted decay '
       'dashed. These are the curves every &lambda; comes from - a straight line here means the decay '
       'really is exponential and the single number is a fair summary.</p>'))
if os.path.exists(f"{FIG}/fig_diffusion_direction.png"):
    B.append(figblock("f-dif2","fig_diffusion_direction","The direction, measured",
       '<p>Left: &lambda; broken out by direction. A <b>circle is an isotropic leak</b> - signal '
       'spreading equally all round, which is diffusion or blur, not transport. A lobe pointing one '
       'way would be the Visium v1 signature. Several platforms are visibly <em>elongated</em>, but '
       'elongated is not the same as one-sided: elongation with no net direction is what tissue '
       'architecture looks like, since ducts and nests are themselves elongated.<br>'
       'Right: the one-sided drift, converted from the raw statistic into the offset in µm that '
       'would produce it. Every bar sits below the red detection floor. Note that the permutation '
       'p-value for these is significant almost everywhere and is <b>deliberately not used as the '
       'criterion</b>: with 10<sup>4</sup>-10<sup>5</sup> bins per section, any trace of spatial '
       'structure is "significant", so the question has to be how big the effect is, not whether it '
       'is non-zero.</p>'))
B.append('</div></section>')

# section 04: cost
B.append('<section class="band" style="background:#fff;border-top:1px solid var(--line);border-bottom:1px solid var(--line)"><div class="wrap">')
B.append('<div class="kicker"><span class="n">04</span><h2>Price per area, and per transcript</h2></div>')
B.append('<p class="sub" style="margin-bottom:18px">Anchored on <b>FGCZ all-in customer prices</b> - what a customer '
         'pays for one billing unit <b>including library prep, sequencing and processing</b> (CHF) - normalised to each '
         'platform&rsquo;s max capture area at full utilisation. All six columns are now all-in FGCZ prices and directly comparable.</p>')
B.append('<div class="grid3">')
B.append(f'<div class="card"><div class="big">CHF {cost_per_100k("stratamap_breast"):.2f}</div><h4>StrataMap &middot; per 100k transcripts (cheapest)</h4>'
         f'<p>Its very high transcript yield makes it the cheapest <b>per transcript</b> here by far. Per <em>area</em> it is also lowest on paper '
         f'(CHF {cost_per_mm2("stratamap_breast"):,.0f}/mm² on the usable {MAX_AREA["stratamap_breast"]:,.0f}&nbsp;mm²), but read that as best-case: '
         'its SBC registration requires strict &gt;1&nbsp;mm gaps between samples, so much of the 624&nbsp;mm² is blank in practice - FGCZ wet-lab experience puts its real per-area cost near Atera. Xenium and Atera are imaging with no such registration limit, so they pack samples tightly and hold their per-area figures.</p></div>')
B.append(f'<div class="card"><div class="big">CHF {cost_per_mm2("visiumhd65_breast_8um"):,.0f}</div><h4>Visium HD &middot; priciest per area</h4>'
         f'<p>Billed per <b>reaction (one capture area)</b>, not per slide - so its small capture areas make it the <b>dearest per mm²</b> here: '
         f'the 6.5&nbsp;mm format is CHF {cost_per_mm2("visiumhd65_breast_8um"):,.0f} / mm², the 11&nbsp;mm CHF {cost_per_mm2("visiumhd11_breast_8um"):,.0f}. '
         f'Its real draw is the <b>lowest-capital sequencing entry point</b> (~CHF {COST["visiumhd65_breast_8um"]["price"]:,.0f} / reaction all-in) and up to 2 capture areas per slide, not per-area value.</p></div>')
B.append(f'<div class="card"><div class="big">CHF {cost_per_100k("prime5k_breast"):.0f}</div><h4>Xenium Prime 5K &middot; priciest per transcript</h4>'
         f'<p>Dearest <b>per transcript</b> here (CHF {cost_per_100k("prime5k_breast"):.0f} / 100k) and pricey per area '
         f'(~CHF {cost_per_mm2("prime5k_breast"):,.0f} / mm², behind only Visium HD): a 5,101-gene panel over a full imaging run reads many genes, none deeply. '
         f'Xenium v1 is cheaper both ways ({cost_per_mm2("stdxenium_breast"):,.0f}/mm², {cost_per_100k("stdxenium_breast"):.1f}/100k).</p></div>')
B.append('</div>')
B.append(figblock("f-cost","10_cost_sensitivity","All-in cost per area: the large-slide sequencer undercuts everyone",
   '<p>Every price here now includes sequencing, so the old &ldquo;imaging is cheap because it skips sequencing&rdquo; logic no longer '
   'holds. Per area, <b>StrataMap&rsquo;s large slide is cheapest</b>, then Visium HD 11&nbsp;mm and Xenium; the targeted <b>Prime 5K is dearest</b>. '
   'With Visium HD billed per <b>slide (2 capture areas)</b> its per-area cost is about half a per-single-area count, and the 11&nbsp;mm format overtakes 6.5&nbsp;mm once well utilised. '
   '<span class="note">FGCZ all-in customer prices for all six platforms (imaging tiers dashed, sequencing tiers as utilisation curves). '
   'CHF/mm² is best-case at full utilisation. StrataMap&rsquo;s SBC registration mandates strict &gt;1&nbsp;mm inter-sample gaps, so much of its '
   'usable strip is blank in practice (real per-area cost near Atera); Xenium and Atera have no strict registration limit and pack samples tightly.</span></p>'))
B.append('</div></section>')

# section 05: per cell type
B.append('<section class="band"><div class="wrap">')
B.append('<div class="kicker"><span class="n">05</span><h2>Where targeted panels matter most: low-RNA cells</h2></div>')
B.append(figblock("f-ct","08_celltype_sensitivity","Sensitivity by cell type (shared genes)",
   '<p>The depth gap widens in RNA-poor cells. In T cells the 313-plex panel still records '
   '<span class="stat">~55</span> shared-gene transcripts/cell while a Visium HD bin sees <span class="stat">2&ndash;3</span>. '
   'If your question is rare immune populations or sparse signalling, a targeted or Atera assay is not a luxury - '
   'it is what makes those cells detectable at all. <span class="note">StrataMap is omitted here: the demo has no cell-type labels and '
   'naive marker typing of it was unreliable (see the StrataMap note); its per-cell shared-marker depth (34) already sits below the '
   'targeted panels regardless of typing.</span></p>'))
B.append('</div></section>')

# section 05b: RCTD on vendor vs Proseg segmentation, like-for-like (40_rctd_matched_rerun.py)
RM = {r["condition"]: r for r in csv.DictReader(open(f"{OUT}/rctd_matched/summary.csv"))}
def _p(c, k): return f'{100*float(RM[c][k]):.1f}%'
B.append('<section class="band" style="background:#fff;border-top:1px solid var(--line);border-bottom:1px solid var(--line)"><div class="wrap">')
B.append('<div class="kicker"><span class="n">05b</span><h2>Cell typing on vendor vs Proseg segmentation (Atera vs StrataMap)</h2></div>')
B.append('<p class="sub" style="margin-bottom:18px">RCTD (rctd-py 0.3.8, doublet mode) on one 1.5&nbsp;&times;&nbsp;1.5&nbsp;mm window per platform, '
         'each segmented two ways: the vendor&rsquo;s cells and Proseg&rsquo;s. Every run uses the same reference genes and the same thresholds, so the '
         'only thing that changes within a platform is the segmentation. <b>Proseg barely moves either platform.</b> What does separate them is '
         'immune calls: under an identical configuration Atera types ' + _p("atera_vendor_fixed","immune_of_singlets") + ' of its singlets as immune, '
         'StrataMap ' + _p("sm_vendor_fixed","immune_of_singlets") + '.</p>')
B.append(figblock("f-rescue","28_rctd_immune_rescue","RCTD spot class and lineage, same window per platform",
    '<p><b>A</b> - Rejects: Atera ' + _p("atera_vendor_fixed","reject") + ' vendor, ' + _p("atera_proseg_fixed","reject") + ' Proseg; '
    'StrataMap grade 1 ' + _p("sm_vendor_fixed","reject") + ' vendor, ' + _p("sm_proseg_fixed","reject") + ' Proseg. '
    '<b>B</b> - Immune share of singlets: Atera ' + _p("atera_vendor_fixed","immune_of_singlets") + ' &rarr; ' + _p("atera_proseg_fixed","immune_of_singlets") +
    '; StrataMap ' + _p("sm_vendor_fixed","immune_of_singlets") + ' &rarr; ' + _p("sm_proseg_fixed","immune_of_singlets") + '. '
    'T/NK singlets: Atera ' + _p("atera_vendor_fixed","T_NK_of_singlets") + ', StrataMap ' + _p("sm_vendor_fixed","T_NK_of_singlets") + '.</p>'))
rows = [("atera_vendor_fixed","Atera","vendor (10x)"),("atera_proseg_fixed","Atera","Proseg"),
        ("sm_vendor_fixed","StrataMap G1","vendor (Illumina)"),("sm_proseg_fixed","StrataMap G1","Proseg"),
        ("sm_vendor_orig","StrataMap G1","vendor, first config*"),("sm_proseg_orig","StrataMap G1","Proseg, first config*")]
t = ('<div class="tablewrap" style="margin-top:20px"><table class="master"><thead><tr><th class="rlab" style="text-align:left">Platform</th>'
     '<th>Segmentation</th><th>Cells</th><th>Rejects</th><th>Singlets</th><th>Doublets</th><th>Immune (of singlets)</th><th>T / NK</th>'
     '<th>Malignant / epithelial</th><th>Stroma</th></tr></thead><tbody>')
for c, plat, seg in rows:
    r = RM[c]
    t += (f'<tr><td class="rlab" style="text-align:left"><b>{plat}</b></td><td>{seg}</td><td class="num">{int(r["n"]):,}</td>'
          + "".join(f'<td class="num">{_p(c,k)}</td>' for k in ["reject","singlet","doublet","immune_of_singlets","T_NK_of_singlets","malignant_of_singlets","stroma_of_singlets"])
          + '</tr>')
t += '</tbody></table></div>'
B.append(t + '<p class="note" style="margin-top:10px">Reference: CELLxGENE Census breast cancer (10,689 cells, 40 types; 10x 3&prime;/5&prime; poly-A, '
         'Census 2025-11-08, restricted to the Atera panel genes; 00_build_rctd_reference.py). Inputs subset to the reference genes; '
         'DOUBLET_THRESHOLD&nbsp;=&nbsp;20&nbsp;&times;&nbsp;3.61 and CONFIDENCE_THRESHOLD&nbsp;=&nbsp;5&nbsp;&times;&nbsp;3.61 for every run; sigma fitted per run. '
         '*First config: the threshold scaling of the first version of this analysis, which multiplied both thresholds by each input&rsquo;s '
         'feature count / 5,000 (12.4 for StrataMap vendor, 7.0 for StrataMap Proseg, 3.6 for Atera). A larger CONFIDENCE_THRESHOLD rejects more '
         'cells, which is where the earlier 55.6% &rarr; 21.2% &ldquo;rescue&rdquo; came from, together with comparing a section-wide vendor subset '
         'against the Proseg window. One window per platform, different specimens and fixation.</p>')
B.append('<div class="grid2" style="margin-top:20px">'
         '<div class="card"><h4>What the window looks like</h4>'
         '<p>In the grade 1 window, Illumina&rsquo;s Expanded-5&nbsp;&micro;m contours leave <b>16.8%</b> of transcripts outside every cell, and Proseg '
         'assigns <b>5.5%</b> to background. Cells with <i>CD3D</i>&nbsp;&ge;&nbsp;1: 108 vendor, 119 Proseg (of ~21.4k); with <i>CD3D</i>&nbsp;&ge;&nbsp;2: '
         '28 vendor, 19 Proseg. Atera&rsquo;s window has 267 vendor / 315 Proseg cells at <i>CD3D</i>&nbsp;&ge;&nbsp;2 out of 12.4k.</p></div>'
         '<div class="card"><h4>What this does and does not say</h4>'
         '<p>Resegmentation does not recover StrataMap&rsquo;s immune cells in this window, and it does not change either platform&rsquo;s reject rate. '
         'The immune gap survives a reference whose poly-A chemistry is closer to StrataMap&rsquo;s. It does not separate chemistry from '
         'specimen: the StrataMap block is fresh-frozen DCIS/IDC grade 1, the Atera block FFPE grade 3.</p></div>'
         '</div>')
B.append('</div></section>')


# section 06: breast vs cervix
B.append('<section class="band" style="background:#fff;border-top:1px solid var(--line);border-bottom:1px solid var(--line)"><div class="wrap">')
B.append('<div class="kicker"><span class="n">06</span><h2>Second tissue: cervix tracks breast</h2></div>')
B.append(figblock("f-bc","09_breast_vs_cervix","Matched high-plex platforms, two cancers",
   '<p>Only Prime 5K and Atera are publicly available for cervical cancer (no standard-panel or Visium HD cervix exists). '
   'Across both tissues the ranking holds: Atera delivers far higher per-gene sensitivity than Prime 5K, confirming the '
   'pattern is a platform property, not a breast-specific quirk. <span class="note">Cervix is a different block, not a serial section.</span></p>'))
B.append('</div></section>')

# section 07: application matrix
B.append('<section class="band"><div class="wrap">')
B.append('<div class="kicker"><span class="n">07</span><h2>Which platform for which application</h2></div>')
B.append('<p class="sub" style="margin-bottom:18px">Synthesis of the metrics above into practical guidance. '
         'Ratings are relative within this benchmark and reflect each platform&rsquo;s specs - Atera is rated on its '
         'data sheet, not current availability (preview, ships H2 2026; high instrument capital).</p>')
B.append(appmatrix())
B.append('<p class="note" style="margin-top:12px">Two calls worth spelling out: for <b>unbiased whole-transcriptome discovery</b>, '
 'only <b>StrataMap</b> is truly unbiased (poly-A captures all transcripts, incl. noncoding and novel) - Atera and Visium HD are '
 '<b>probe-based</b>, so they profile a fixed ~18k protein-coding panel comprehensively but cannot discover what is not on the panel '
 '(hence "Good", not "Best"). For <b>large area / many samples</b>, <b>Atera</b> (4 slides/run, ~2,000 mm²), <b>Xenium</b> (2 slides, ~470 mm²) and '
 '<b>StrataMap</b> all rate Best - StrataMap for its large single slide (624 mm² usable) that holds <em>multiple</em> sections at the lowest cost per '
 'transcript (an earlier "Good" assumed it ran one sample/slide, since corrected). Visium HD is billed <b>per reaction</b> (one 42/121 mm² capture area, 2 per slide), so its small, dear capture areas make it the weakest large-area-on-budget option here; '
 'all sequencing-readout columns are ultimately sequencing-capacity-limited, not slide-limited.</p>')
B.append('</div></section>')

# methods
B.append('<section class="band" style="padding-top:8px"><div class="wrap">')
B.append('<details class="methods"><summary>Methods, data sources &amp; caveats</summary>')
B.append('<h4>Datasets</h4><p>All 10x public FFPE. Breast: standard Xenium (Janesick, 313 GEX genes), Xenium Prime 5K '
 '(5,101), Atera whole-transcriptome preview (18,028), Visium HD 6.5&nbsp;mm (SR 3.1.2) and 11&nbsp;mm TMA (SR 4.1.0), '
 'both probe set v2 (~18,085 genes); fresh-frozen Visium HD 6.5&nbsp;mm (SR 4.0.1, probe set v2) for the prep contrast. Cervix: '
 'Prime 5K + Atera only. These are <b>different blocks</b>, all human cancer FFPE - not serial sections, so absolute differences '
 'include some section/RNA-quality variation, not chemistry alone. <b>Dissociated scRNA-seq reference</b> (breast cancer): 10x 3&prime; '
 'v3, 5&prime; v2 and Smart-seq2 from CELLxGENE Census (median genes/UMIs per cell from raw counts, ~3k cells/assay subsampled), '
 'plus a deeply-sequenced 10x Flex (probe-based) breast sample (the public 4-plex dissociated-tumour-cell + TotalSeq-C set, '
 'CellRanger 7.2.0; ~36k UMIs/cell) - all whole-transcriptome, strongly sequencing-depth-dependent. '
 '<b>Cross-vendor vignette:</b> CosMx Human Multiomic Breast (Bruker/NanoString SMI, public FFPE section; WTx panel 18,942 genes + '
 '64-plex protein) - recomputed identically from the RNA <code>exprMat</code> flat file (152,451 cells); negative probes and '
 'SystemControl/falsecode codewords are its two control classes, mapped to Atera&rsquo;s negative-control probes and codewords. '
 '<b>Illumina StrataMap</b> (NovaSeq X, 10&nbsp;µm fresh-frozen human breast cancer, IDC grade 3, DRAGEN Spatial Transcriptome workflow; '
 'imported from BaseSpace Sequence Hub Data Central via the BaseSpace CLI) ships a <b>raw 1&nbsp;µm spatial-barcode (SBC) matrix</b> '
 '(61,906 gene models &times; 44.8&nbsp;M SBCs, 2.6&nbsp;billion nonzeros), not a cell matrix. It is recomputed <b>segmentation-free</b>: '
 'gene pseudobulk over the full matrix, area = distinct 8&nbsp;µm-occupied bins &times; 64&nbsp;µm² (SBC coordinates are encoded as '
 'Y:X&nbsp;nm in the barcodes), giving transcripts/mm² and transcripts/mm²/gene on the shared sets - the same intensive, segmentation-free '
 'definition used elsewhere. Its <b>per-cell and cost figures are intentionally absent</b> (a raw SBC matrix has no cells; poly(A) chemistry '
 'and 7.5&nbsp;cm² capture are vendor specs), so StrataMap appears only on the breadth / per-area / per-gene axes, never in the per-cell or '
 'cost tiers.</p>')
B.append('<h4>Metric recompute</h4><p>Every number is recomputed from <code>cell_feature_matrix.h5</code> (imaging) or the '
 '8&nbsp;µm-bin <code>filtered_feature_bc_matrix.h5</code> (Visium HD; the 11&nbsp;mm per-cell figures use its SpaceRanger&nbsp;4.1 '
 '<code>filtered_feature_cell_matrix.h5</code> segmentation, 6.5&nbsp;mm is binned-only), not copied from vendor <code>metrics_summary.csv</code> '
 '(which are not comparable across chemistries). Per-gene sensitivity = total shared-gene transcripts &divide; area &divide; gene count, '
 'which removes the cell-vs-bin size confound. Shared sets: <b>L1 = 304</b> = the 313-plex flagship panel genes carried by the '
 'whole-transcriptome comparators (Atera + both Visium HD; CosMx 311/313 and StrataMap 313/313 also carry them), resolved across '
 'HGNC symbol renames - the targeted Prime 5K (187/313) is scored in L2 instead, not allowed to shrink the core; '
 '<b>L2 = 4,843</b> genes across the four high-plex tiers (Prime 5K, Atera, both Visium HD). Specificity = negative-control-probe counts as a fraction of '
 'gene-expression counts (imaging). Concordance = Spearman on CPM-normalised pseudobulk over shared genes. '
 'Cell types = marker-score argmax (approximate; flagged).</p>')
B.append('<h4>Pricing model</h4><p><b>FGCZ all-in customer prices</b> (CHF) - what a customer pays per billing unit '
 '<b>including library prep, sequencing and processing</b> (imaging tiers need no sequencing): Xenium v1 <b>CHF 13,321 / run</b> (2 slides); '
 'Xenium Prime <b>CHF 21,782 / run</b> (2 slides); Atera <b>CHF 29,400 / run</b> (2 slides); Visium HD 6.5&nbsp;mm <b>CHF 3,862 / reaction</b>; '
 'Visium HD 11&nbsp;mm <b>CHF 5,735 / reaction</b>; StrataMap <b>CHF 9,385 / slide</b>. '
 'Atera is a new instrument (~500&nbsp;mm²/slide, up to 4 slides/run, ships H2 2026); its price is an FGCZ estimate quoted per 2 slides. '
 '<b>Capture area</b> - per slide (× slides/run = max throughput): Xenium 235 × 2 = 470, Atera 500 × 4 = 2,000, '
 'Visium HD billed per reaction = 1 capture area (42.25 / 121 mm²; a slide carries 2), StrataMap usable 13×48 = 624 mm² (of a 7.5&nbsp;cm² flow cell; holds multiple sections). '
 'StrataMap&rsquo;s SBC registration also mandates strict &gt;1&nbsp;mm inter-sample gaps (Xenium/Atera imaging have no such limit and pack tighter), so its per-area figure is especially best-case. CHF/mm² and '
 'CHF/100k-transcripts are computed at <b>full capture-area utilisation</b> (best case) using each platform&rsquo;s '
 'measured transcript density; sensitivity metrics are normalised to the actually analysed tissue area. '
 'Figures are estimates, not quotes; USD&asymp;CHF at ~1:1.</p>')
B.append('<h4>Reproducibility</h4><p>Pipeline + cached intermediates at '
 '<code>/srv/GT/analysis/pgueguen/spatial_platform_comparison</code>.</p>')
B.append('</details></div></section>')

B.append('<footer><div class="wrap">FGCZ &middot; spatial platform benchmark &middot; recomputed '
         'from 10x public FFPE matrices &middot; figures and pricing are estimates, see methods.</div></footer>')
B.append('</div>')  # .page

html = ('<meta charset="utf-8">\n'
        '<title>Xenium vs Atera vs Visium HD vs StrataMap - spatial platform benchmark</title>\n'
        '<meta name="description" content="Sensitivity, specificity and price-per-area for six spatial transcriptomics assays '
        '(10x Xenium/Atera/Visium HD, Illumina StrataMap, + CosMx), recomputed on a shared gene set from matched human breast cancer.">\n'
        + CSS + "\n" + "".join(B))

# house style: no em/en dashes in delivered prose (hyphens only)
html = (html.replace("&mdash;"," - ").replace("&ndash;","-")
            .replace("—"," - ").replace("–","-"))

for path in [f"{SCRATCH}/spatial_platform_comparison.html", f"{OUT}/spatial_platform_comparison.html"]:
    with open(path,"w") as f: f.write(html)
print("HTML bytes:",len(html))
print("written:",f"{SCRATCH}/spatial_platform_comparison.html")
