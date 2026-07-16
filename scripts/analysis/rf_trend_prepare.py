"""Assemble matrix, define the fixed injection (boost), create train/test split."""
import json, glob
import numpy as np

OUT="/tmp/rf_trend"
names=json.load(open(f"{OUT}/colnames.json"))
parts=sorted(glob.glob(f"{OUT}/part_*.npy"))
X=np.vstack([np.load(p) for p in parts]).astype(np.float32)
print("matrix:", X.shape, f"{X.nbytes/1e6:.0f} MB")
ci={n:i for i,n in enumerate(names)}
y=X[:,ci["Is Laundering"]].astype(np.int8)
print("positives:", int(y.sum()), "| negatives:", int((y==0).sum()))

# log-transform amounts (as in pipeline)
for c in ["Amount Received","Amount Paid","src_day_tx_count","dst_day_tx_count"]:
    X[:,ci[c]]=np.log1p(np.maximum(X[:,ci[c]],0))

# RAT_score among laundering rows -> injection thresholds (as in injector)
rs=X[y==1, ci["RAT_score"]]
thr={lvl: float(np.quantile(rs, 1-f)) for lvl,f in [("low",.15),("medium",.30),("high",.60)]}
print("thresholds:", {k:round(v,4) for k,v in thr.items()})

np.save(f"{OUT}/X.npy", X); np.save(f"{OUT}/y.npy", y)
json.dump({"thr":thr, "ci":ci}, open(f"{OUT}/meta.json",'w'))

# fixed stratified 80/20 split
rng=np.random.RandomState(42)
idx=np.arange(len(y)); rng.shuffle(idx)
pos=idx[y[idx]==1]; neg=idx[y[idx]==0]
te=np.concatenate([pos[:len(pos)//5], neg[:len(neg)//5]])
tr=np.concatenate([pos[len(pos)//5:], neg[len(neg)//5:]])
np.save(f"{OUT}/tr.npy",tr); np.save(f"{OUT}/te.npy",te)
print("train:",len(tr),"test:",len(te),"| test positives:", int(y[te].sum()))
# cleanup parts
import os
for p in parts: os.remove(p)
print("OK")
