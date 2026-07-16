"""Resumable column extractor: RAT CSV -> compact float32 matrix.
Keeps ALL laundering rows; samples ~30% of negatives deterministically."""
import os, io, json, sys
import numpy as np
import pandas as pd

CSV="/sessions/nice-nifty-meitner/mnt/Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection/ibm_transcations_datasets/RAT/HI-Small_Trans_RAT_medium.csv"
OUT="/tmp/rf_trend"
STATE=f"{OUT}/state.json"
BLOCK=350_000_000  # bytes per block
TIME_BUDGET=33     # seconds

# columns (0-based): label, amounts, time, all RAT/motif numerics + injected flag
COLS = {
 "Amount Received":5, "Amount Paid":7, "Is Laundering":10,
 "src_age_days":19, "dst_age_days":20, "src_day_tx_count":22, "dst_day_tx_count":23,
 "hour":24, "weekday":25,
 "RAT_is_off_hours":26, "RAT_is_weekend":27, "RAT_is_cross_bank":28,
 "RAT_src_amount_z_pos":29, "RAT_dst_amount_z_pos":30,
 "RAT_src_out_deg_norm":31, "RAT_dst_in_deg_norm":32,
 "RAT_src_burst_norm":33, "RAT_dst_burst_norm":34, "RAT_combined_burst":35,
 "RAT_same_entity":46, "RAT_src_entity_acct_norm":49, "RAT_dst_entity_acct_norm":50,
 "RAT_src_pattern_flag":51, "RAT_dst_pattern_flag":52, "RAT_mutual_flag":53,
 "dst_out_deg_norm":55,
 "motif_fanin":56, "motif_fanout":57, "motif_chain":58, "motif_cycle":59,
 "RAT_offender_score":60, "RAT_target_score":61, "RAT_guardian_weakness_score":62,
 "RAT_score":63, "RAT_injected":64,
}
names=list(COLS); idxs=[COLS[n] for n in names]

import time
t0=time.time()
state={"offset":0,"part":0,"done":False}
if os.path.exists(STATE): state=json.load(open(STATE))
if state["done"]: print("ALREADY DONE"); sys.exit()

sz=os.path.getsize(CSV)
fh=open(CSV,'rb')
rng_mod=10  # keep negatives where hash%10<3 => 30%

while time.time()-t0 < TIME_BUDGET and state["offset"] < sz:
    fh.seek(state["offset"])
    blob=fh.read(BLOCK)
    if state["offset"]==0:
        nl=blob.index(b'\n'); blob_body=blob[nl+1:]
        consumed_start=nl+1
    else:
        blob_body=blob; consumed_start=0
    # cut at last full line
    last_nl=blob_body.rfind(b'\n')
    if last_nl==-1: break
    body=blob_body[:last_nl]
    consumed=consumed_start+last_nl+1
    df=pd.read_csv(io.BytesIO(body), header=None, usecols=idxs,
                   names=[f"c{i}" for i in range(66)], engine='c',
                   na_filter=True, low_memory=False)
    df.columns=[names[idxs.index(int(c[1:]))] for c in df.columns]
    df=df.apply(pd.to_numeric, errors='coerce').astype(np.float32).fillna(0)
    y=df["Is Laundering"].values
    n=len(df)
    # deterministic negative sampling by global row index
    start_row=state.get("rows_seen",0)
    gidx=np.arange(start_row, start_row+n)
    keep=(y==1)|((gidx % rng_mod)<3)
    arr=df.values[keep]
    np.save(f"{OUT}/part_{state['part']:03d}.npy", arr)
    state["part"]+=1
    state["offset"]+=consumed
    state["rows_seen"]=start_row+n
    json.dump(state,open(STATE,'w'))
    print(f"block {state['part']}: rows {n}, kept {keep.sum()}, offset {state['offset']/1e9:.2f}/{sz/1e9:.2f} GB, t={time.time()-t0:.0f}s")

if state["offset"]>=sz:
    state["done"]=True
    json.dump(state,open(STATE,'w'))
    json.dump(names, open(f"{OUT}/colnames.json",'w'))
    print("EXTRACTION COMPLETE")
