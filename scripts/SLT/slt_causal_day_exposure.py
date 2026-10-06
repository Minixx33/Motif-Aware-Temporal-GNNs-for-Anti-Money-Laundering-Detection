"""
slt_causal_day_exposure.py
--------------------------
Point-in-time (causal) version of the CURRENT-DAY SLT exposure features used by
slt_injector.py:

    SLT_{src,dst}_susp_nbr_ratio, SLT_{src,dst}_susp_amt_share,
    SLT_{src,dst}_susp_txn_share, SLT_{src,dst}_strong_tie_susp_ratio

The original injector aggregated each account's exposure over the WHOLE
calendar day and merged that onto every transaction of the day, so a 09:00
transaction saw counterparties/amounts from the account's 23:00 transactions
(future information; no labels, but look-ahead). Here every value is the
account's same-day exposure from transactions STRICTLY BEFORE the current row
(earlier row position on the same calendar day -- the same "before" convention
as every other causal feature in rat_injector.py / slt_injector.py; rows are in
chronological order). The account's exposure combines both roles, exactly like
the original day_exposure table:

  per (account, role, peer, day) "pair":
      pair_txn_count, pair_amount (paid as sender / received as receiver),
      peer_is_high_risk = max of the per-transaction causal high-risk flag
      is_strong_tie     = pair_txn_count > 1  or  pair_amount > median[role]
  per (account, day), summed over pairs:
      neighbors, high_risk_neighbors, tie_amount, high_risk_tie_amount,
      ties, high_risk_ties, strong_ties, strong_ties_high_risk
  features = high_risk_x / x  (0 when x == 0)

All pair quantities are the running values up to (and excluding) the current
row, so with every transaction of the day in the past the formulas reduce
exactly to the original full-day definitions.

Self-test (brute force, synthetic + optional real CSV slice):
    python scripts/SLT/slt_causal_day_exposure.py --selftest [--csv path --nrows 200000]
"""

import numpy as np
import pandas as pd

FEATURES = ("susp_nbr_ratio", "susp_amt_share", "susp_txn_share", "strong_tie_susp_ratio")


def _ratio(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = np.zeros_like(a)
    nz = b > 0
    out[nz] = a[nz] / b[nz]
    return np.clip(out, 0, 1).astype(np.float32)


def causal_same_day_exposure(src, dst, day, amt_paid, amt_rec, src_hr, dst_hr,
                             median_src_role, median_dst_role):
    """
    All inputs are 1-D arrays aligned with the transaction rows, rows in
    chronological order:
      src, dst          integer account codes
      day               integer calendar-day id
      amt_paid, amt_rec amounts
      src_hr, dst_hr    0/1 causal high-risk-peer flags of the src / dst account
                        on this transaction (slt_injector's *_is_high_risk_peer)
      median_*_role     strong-tie amount thresholds (pair-day amount medians)
    Returns {"SLT_src_<feat>": array, "SLT_dst_<feat>": array} for the 4 features.
    """
    n = len(src)
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    day = np.asarray(day, dtype=np.int64)
    orig = np.arange(n, dtype=np.int64)

    # touches: (row, role 0 = as sender, role 1 = as receiver), ordered by row then role
    acct = np.empty(2 * n, np.int64); acct[0::2] = src; acct[1::2] = dst
    peer = np.empty(2 * n, np.int64); peer[0::2] = dst; peer[1::2] = src
    role = np.tile(np.array([0, 1], np.int64), n)
    tday = np.repeat(day, 2)
    amt = np.empty(2 * n, np.float64); amt[0::2] = amt_paid; amt[1::2] = amt_rec
    phr = np.empty(2 * n, np.int8); phr[0::2] = np.asarray(dst_hr); phr[1::2] = np.asarray(src_hr)
    med = np.where(role == 0, float(median_src_role), float(median_dst_role))

    t = pd.DataFrame({"acct": acct, "role": role, "peer": peer, "day": tday,
                      "amt": amt, "phr": phr})
    pair = t.groupby(["acct", "role", "peer", "day"], sort=False)
    pc = pair.cumcount().to_numpy() + 1                     # pair count after this touch
    pa = pair["amt"].cumsum().to_numpy()                     # pair amount after
    ph = pair["phr"].cummax().to_numpy().astype(np.int8)     # pair high-risk after
    t["ph"] = ph
    ph_prev = (t.groupby(["acct", "role", "peer", "day"], sort=False)["ph"]
               .shift(1).fillna(0).to_numpy().astype(np.int8))
    pc_prev = pc - 1
    pa_prev = pa - amt

    strong = (pc > 1) | (pa > med)
    strong_prev = (pc_prev > 1) | ((pc_prev > 0) & (pa_prev > med))
    sh = strong & (ph == 1)
    sh_prev = strong_prev & (ph_prev == 1)
    becomes_hr = (ph == 1) & (ph_prev == 0)
    was_hr = ph_prev == 1

    inc = {
        "nbr": (pc == 1).astype(np.float64),
        "hr_nbr": becomes_hr.astype(np.float64),
        "amt": amt,
        "hr_amt": np.where(was_hr, amt, np.where(becomes_hr, pa, 0.0)),
        "ties": np.ones(2 * n),
        "hr_ties": np.where(was_hr, 1.0, np.where(becomes_hr, pc, 0.0)),
        "strong": (strong & ~strong_prev).astype(np.float64),
        "strong_hr": (sh & ~sh_prev).astype(np.float64),
    }

    # running account-day totals AFTER each touch, then remove this row's own
    # touches (both of them for a self-loop) -> state strictly before the row
    ad_keys = [t["acct"], t["day"]]
    selfloop = np.repeat(src == dst, 2)
    is_recv = role == 1
    before = {}
    for k, v in inc.items():
        after = pd.Series(v).groupby(ad_keys, sort=False).cumsum().to_numpy()
        own = v.copy()
        # receiver-role touch of a self-loop row: also remove the sender-role touch
        prev_touch = np.r_[0.0, v[:-1]]
        own = np.where(is_recv & selfloop, v + prev_touch, own)
        before[k] = after - own

    out = {}
    for r, name in ((0, "src"), (1, "dst")):
        sel = role == r
        b = {k: v[sel] for k, v in before.items()}
        out[f"SLT_{name}_susp_nbr_ratio"] = _ratio(b["hr_nbr"], b["nbr"])
        out[f"SLT_{name}_susp_amt_share"] = _ratio(b["hr_amt"], b["amt"])
        out[f"SLT_{name}_susp_txn_share"] = _ratio(b["hr_ties"], b["ties"])
        out[f"SLT_{name}_strong_tie_susp_ratio"] = _ratio(b["strong_hr"], b["strong"])
    return out


# ---------------------------------------------------------------------------
# brute-force reference + self-test
# ---------------------------------------------------------------------------

def _brute(i, src, dst, day, amt_paid, amt_rec, src_hr, dst_hr, med_s, med_d, side):
    """Exposure of the side-account of row i from same-day rows < i (direct definition)."""
    A = src[i] if side == "src" else dst[i]
    pairs = {}
    for j in range(i):
        if day[j] != day[i]:
            continue
        for role, a, p, amt, hr in ((0, src[j], dst[j], amt_paid[j], dst_hr[j]),
                                    (1, dst[j], src[j], amt_rec[j], src_hr[j])):
            if a != A:
                continue
            key = (role, p)
            c, s, h = pairs.get(key, (0, 0.0, 0))
            pairs[key] = (c + 1, s + amt, max(h, int(hr)))
    nbr = len(pairs)
    hr_nbr = sum(h for (_, _, h) in pairs.values())
    tot_amt = sum(s for (_, s, _) in pairs.values())
    hr_amt = sum(s * h for (_, s, h) in pairs.values())
    ties = sum(c for (c, _, _) in pairs.values())
    hr_ties = sum(c * h for (c, _, h) in pairs.values())
    strong = {k: (c > 1 or s > (med_s if k[0] == 0 else med_d)) for k, (c, s, _) in pairs.items()}
    st = sum(strong.values())
    st_hr = sum(1 for k, (c, s, h) in pairs.items() if strong[k] and h)
    r = lambda a, b: min(max(a / b, 0), 1) if b > 0 else 0.0
    return {"susp_nbr_ratio": r(hr_nbr, nbr), "susp_amt_share": r(hr_amt, tot_amt),
            "susp_txn_share": r(hr_ties, ties), "strong_tie_susp_ratio": r(st_hr, st)}


def _check(src, dst, day, ap, ar, sh, dh, ms, md, n_checks, rng, label):
    out = causal_same_day_exposure(src, dst, day, ap, ar, sh, dh, ms, md)
    idx = rng.choice(len(src), size=min(n_checks, len(src)), replace=False)
    for i in idx:
        for side in ("src", "dst"):
            ref = _brute(i, src, dst, day, ap, ar, sh, dh, ms, md, side)
            for f in FEATURES:
                got = float(out[f"SLT_{side}_{f}"][i])
                assert abs(got - ref[f]) < 1e-5, f"{label}: row {i} {side} {f}: {got} vs {ref[f]}"
    # future invariance: changing rows >= cut must not change rows < cut
    cut = len(src) // 2
    ap2 = ap.copy(); ap2[cut:] *= 7.0
    sh2 = sh.copy(); sh2[cut:] = 1 - sh2[cut:]
    out2 = causal_same_day_exposure(src, dst, day, ap2, ar, sh2, dh, ms, md)
    for k in out:
        assert np.array_equal(out[k][:cut], out2[k][:cut]), f"{label}: future rows changed {k}"
    print(f"[{label}] {len(idx)} rows x 2 sides x 4 features match brute force; "
          f"future-row invariance OK")
    return out


def _selftest(csv=None, nrows=200_000, n_checks=300):
    rng = np.random.default_rng(0)
    # synthetic: few accounts -> many repeats, self-loops, both roles, multi-day
    n = 3000
    src = rng.integers(0, 40, n); dst = rng.integers(0, 40, n)
    loops = rng.random(n) < 0.05; dst[loops] = src[loops]
    day = np.sort(rng.integers(0, 4, n))
    ap = rng.lognormal(5, 2, n); ar = ap * rng.uniform(0.9, 1.1, n)
    sh = (rng.random(n) < 0.3).astype(np.int8); dh = (rng.random(n) < 0.3).astype(np.int8)
    _check(src, dst, day, ap, ar, sh, dh, 150.0, 200.0, n_checks, rng, "synthetic")

    if csv:
        df = pd.read_csv(csv, nrows=nrows, low_memory=False)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
        codes, _ = pd.factorize(pd.concat([df["Account"].astype(str), df["Account.1"].astype(str)]))
        s, d = codes[:len(df)], codes[len(df):]
        dd = df["Timestamp"].dt.floor("D").astype("int64") // 86_400_000_000_000
        r = np.random.default_rng(1)
        sh = (r.random(len(df)) < 0.1).astype(np.int8); dh = (r.random(len(df)) < 0.1).astype(np.int8)
        # brute force is O(rows) per check -> restrict checks to the first 20k rows
        m = min(len(df), 20_000)
        _check(s[:m], d[:m], dd.to_numpy()[:m], df["Amount Paid"].to_numpy(float)[:m],
               df["Amount Received"].to_numpy(float)[:m], sh[:m], dh[:m],
               float(np.median(df["Amount Paid"])), float(np.median(df["Amount Received"])),
               n_checks, r, f"real CSV first {m:,} rows")
        # full slice through the vectorised path (speed / no crash)
        import time
        t0 = time.time()
        causal_same_day_exposure(s, d, dd.to_numpy(), df["Amount Paid"].to_numpy(float),
                                 df["Amount Received"].to_numpy(float), sh, dh, 1e4, 1e4)
        print(f"[real CSV {len(df):,} rows] vectorised pass in {time.time() - t0:.1f}s")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--nrows", type=int, default=200_000)
    a = ap.parse_args()
    if a.selftest:
        _selftest(a.csv, a.nrows)
