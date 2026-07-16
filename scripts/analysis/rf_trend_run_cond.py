"""Train RF for one condition (none/low/medium/high) with resumable warm_start.
Usage: run_cond.py <condition> [seed]"""
import sys, os, json, time, pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

t0=time.time(); BUDGET=30
cond=sys.argv[1]; seed=int(sys.argv[2]) if len(sys.argv)>2 else 0
OUT="/tmp/rf_trend"
meta=json.load(open(f"{OUT}/meta.json")); ci=meta["ci"]; thr=meta["thr"]
X=np.load(f"{OUT}/X.npy"); y=np.load(f"{OUT}/y.npy")
tr=np.load(f"{OUT}/tr.npy"); te=np.load(f"{OUT}/te.npy")

ALPHA=0.7
BOOST=["RAT_src_amount_z_pos","RAT_dst_amount_z_pos","RAT_src_out_deg_norm","RAT_dst_in_deg_norm",
       "RAT_src_burst_norm","RAT_dst_burst_norm","RAT_combined_burst",
       "RAT_src_entity_acct_norm","RAT_dst_entity_acct_norm",
       "motif_fanin","motif_fanout","motif_chain","motif_cycle"]

def q95(col): return float(np.quantile(X[:,ci[col]],0.95))

Xc=X.copy()
if cond!="none":
    inj=(y==1)&(X[:,ci["RAT_score"]]>=thr[cond])
    print(f"{cond}: boosting {inj.sum()} of {int(y.sum())} laundering rows")
    for c in BOOST:
        hi=max(q95(c), Xc[inj,ci[c]].max()*0)  # q95 target
        cur=Xc[inj,ci[c]]
        Xc[inj,ci[c]]=np.maximum(cur, cur+ALPHA*(hi-cur)).astype(np.float32)
    # recompute composites (injector formulas)
    g=lambda c: Xc[:,ci[c]]
    age_norm=np.clip(g("dst_age_days")/ (np.quantile(X[:,ci["dst_age_days"]],0.95)+1e-8),0,1)
    off=( .30*g("RAT_src_amount_z_pos")+.20*g("RAT_src_out_deg_norm")+.20*g("RAT_src_burst_norm")
         +.10*g("RAT_is_off_hours")+.10*g("RAT_src_pattern_flag")+.10*g("RAT_src_entity_acct_norm"))
    tar=( .35*g("RAT_dst_amount_z_pos")+.25*g("RAT_dst_in_deg_norm")+.15*(1-age_norm)
         +.15*g("RAT_dst_entity_acct_norm")+.10*g("RAT_dst_pattern_flag"))
    gua=( .30*g("RAT_is_off_hours")+.20*g("RAT_is_weekend")+.20*g("RAT_is_cross_bank")
         +.20*g("RAT_combined_burst")+.10*g("RAT_same_entity"))
    Xc[:,ci["RAT_offender_score"]]=off
    Xc[:,ci["RAT_target_score"]]=tar
    Xc[:,ci["RAT_guardian_weakness_score"]]=gua
    Xc[:,ci["RAT_score"]]=np.clip(((off+1e-8)*(tar+1e-8)*(gua+1e-8))**(1/3),0,1)

# model-input features: exclude label, injected flag, raw ages
EXCL={"Is Laundering","RAT_injected","src_age_days","dst_age_days"}
feats=[n for n in ci if n not in EXCL]
fidx=[ci[n] for n in feats]
Xtr,ytr=Xc[tr][:,fidx],y[tr]; Xte,yte=Xc[te][:,fidx],y[te]

TOTAL_TREES=60; BATCH=10
ckpt=f"{OUT}/rf_{cond}_s{seed}.pkl"
if os.path.exists(ckpt):
    clf=pickle.load(open(ckpt,'rb'))
else:
    clf=RandomForestClassifier(n_estimators=0, warm_start=True, max_depth=14,
        max_samples=0.3, bootstrap=True, n_jobs=2, random_state=seed,
        min_samples_leaf=2)
while clf.n_estimators<TOTAL_TREES and time.time()-t0<BUDGET:
    clf.n_estimators+=BATCH
    clf.fit(Xtr,ytr)
    pickle.dump(clf,open(ckpt,'wb'))
    print(f"trees={clf.n_estimators} t={time.time()-t0:.0f}s")
if clf.n_estimators>=TOTAL_TREES:
    p=clf.predict_proba(Xte)[:,1]
    aupr=average_precision_score(yte,p)
    res=json.load(open(f"{OUT}/results.json")) if os.path.exists(f"{OUT}/results.json") else {}
    res[f"{cond}_s{seed}"]=aupr
    json.dump(res,open(f"{OUT}/results.json",'w'))
    print(f"DONE {cond} seed{seed}: test AUPR={aupr:.4f}")
else:
    print(f"PARTIAL {cond}: {clf.n_estimators}/{TOTAL_TREES} trees")
