#!/usr/bin/env python3
"""Reconcile the StrataMap alignment reference (GENCODE ~v41 / GRCh38.p13, 61,906 gene models) with the
current FGCZ p14 reference (GENCODE Release 48, 2025-07-03). Question: does the version gap change the
gene set the cross-platform comparison relies on, or is it just symbol drift (already handled by aliases)?
Compares on version-stripped ENSG. Reports: stable / retired / gained, symbol drift, and stability of the
protein-coding set + the 304-gene shared core."""
import gzip, json, glob
ROOT="/srv/GT/analysis/pgueguen/spatial_platform_comparison"; OUT=f"{ROOT}/outputs"
P14="/srv/GT/reference/Homo_sapiens/GENCODE/GRCh38.p14/Annotation/Release_48-2025-07-03/Genes/features_annotation_byGene.txt"
SMF="/srv/GT/analysis/spatial_platform_comparison".replace("/analysis/","/analysis/pgueguen/")+"/data/stratamap_breast/grade3/Raw_Matrix_Files/features.tsv.gz"

# --- p14 (GENCODE v48): ENSG (already stripped) -> (symbol, biotype) ---
p14={}
with open(P14) as fh:
    next(fh)
    for line in fh:
        p=line.rstrip("\n").split("\t")
        p14[p[0].split(".")[0]]=(p[2], p[3])
print(f"p14 (GENCODE v48): {len(p14):,} genes")

# --- StrataMap (~v41): versioned ENSG -> symbol ---
sm={}   # stripped ENSG -> (symbol, full_versioned_id)
with gzip.open(SMF,"rt") as fh:
    for line in fh:
        q=line.rstrip("\n").split("\t")
        if len(q)<2: continue
        eid=q[0].split(".")[0]
        if eid.startswith("ENSG"): sm[eid]=(q[1].upper(), q[0])
print(f"StrataMap (~v41):  {len(sm):,} gene models")

# --- overlap on ENSG ---
sm_ids=set(sm); p14_ids=set(p14)
stable = sm_ids & p14_ids           # StrataMap genes still present in p14 (same ENSG)
retired= sm_ids - p14_ids           # in StrataMap, gone from p14 (retired/merged)
gained = p14_ids - sm_ids           # new in p14, not in StrataMap
# symbol drift among stable
drift=[]
for e in stable:
    a=sm[e][0]; b=p14[e][0].upper()
    if a!=b: drift.append((e,a,b))
print(f"\nENSG reconciliation (version-stripped):")
print(f"  stable  (StrataMap ENSG present in p14): {len(stable):,}  ({100*len(stable)/len(sm):.1f}% of StrataMap)")
print(f"  retired (StrataMap ENSG gone in p14):    {len(retired):,}  ({100*len(retired)/len(sm):.1f}%)")
print(f"  gained  (new in p14, absent StrataMap):  {len(gained):,}")
print(f"  symbol drift among stable:               {len(drift):,}  (e.g. {drift[:5]})")

# --- protein-coding stability (the set that feeds the '18k genes' / bulk comparison) ---
pc_ensg=set(x.split(".")[0] for x in open(f"{ROOT}/data/stratamap_breast/pc_ensg.txt").read().split())
pc_stable=pc_ensg & p14_ids
pc_pc_in_p14={e for e in pc_stable if p14[e][1]=="protein_coding"}
print(f"\nProtein-coding set (StrataMap PC list = {len(pc_ensg):,}):")
print(f"  present in p14:                {len(pc_stable):,}  ({100*len(pc_stable)/len(pc_ensg):.1f}%)")
print(f"  still typed protein_coding p14:{len(pc_pc_in_p14):,}  ({100*len(pc_pc_in_p14)/len(pc_ensg):.1f}%)")

# --- 304-gene shared core: are they stable across both refs (by symbol, with alias) ---
ALIAS={"KARS1":"KARS","LARS1":"LARS","NARS1":"NARS","QARS1":"QARS","WARS1":"WARS","CYRIA":"FAM49A"}
core=set(open(f"{OUT}/shared_genes_L1.txt").read().split())
p14_syms={ALIAS.get(v[0].upper(),v[0].upper()) for v in p14.values()}
sm_syms ={ALIAS.get(v[0],v[0]) for v in sm.values()}
core_in_p14 = {g for g in core if g in p14_syms or ALIAS.get(g,g) in p14_syms}
core_in_sm  = {g for g in core if g in sm_syms  or ALIAS.get(g,g) in sm_syms}
print(f"\n304-gene shared core (symbol match, alias-aware):")
print(f"  present in p14 (v48): {len(core_in_p14)}/{len(core)}")
print(f"  present in StrataMap: {len(core_in_sm)}/{len(core)}")
print(f"  core genes missing from p14: {sorted(core - core_in_p14)}")

stats=dict(
  p14_release="GENCODE Release 48 (2025-07-03, GRCh38.p14)", p14_genes=len(p14),
  stratamap_ref="GENCODE ~v41 (GRCh38.p13)", stratamap_models=len(sm),
  ensg_stable=len(stable), ensg_stable_pct=round(100*len(stable)/len(sm),1),
  ensg_retired=len(retired), ensg_gained=len(gained), symbol_drift=len(drift),
  pc_list=len(pc_ensg), pc_present_p14=len(pc_stable), pc_present_p14_pct=round(100*len(pc_stable)/len(pc_ensg),1),
  pc_still_pc_p14=len(pc_pc_in_p14),
  core_n=len(core), core_in_p14=len(core_in_p14), core_in_stratamap=len(core_in_sm),
  core_missing_p14=sorted(core - core_in_p14),
  drift_examples=[{"ensg":e,"stratamap":a,"p14":b} for e,a,b in sorted(drift)[:20]])
json.dump(stats, open(f"{OUT}/reconcile_ref_stats.json","w"), indent=2)
print("\nwrote reconcile_ref_stats.json")
