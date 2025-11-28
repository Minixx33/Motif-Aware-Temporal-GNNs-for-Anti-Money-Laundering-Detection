"""
strain_injector_ibm_small.py

Strain Theory injector for the IBM HI-Small AMLSim dataset.

Input  (must be in SAME folder as this script):
    HI-Small_Trans.csv
    HI-Small_accounts.csv
    (HI-Small_Patterns.txt is not required here)

Output (created in ./strain/):
    HI-Small_accounts_STRAIN.csv
    HI-Small_Trans_STRAIN.csv

Account-level features (added):
    Account_ID
    NetFlow_7d
    NetFlow_30d
    NetFlow_90d
    OutInRatio_30d
    IncomingVolatility_30d
    LiquidityShockCount_30d
    TopPartnerConcentration
    BankDependenceScore
    NetworkCentrality
    NetworkStrainExposure
    ChronicStrainScore

Transaction-level features (added):
    TxID
    OutgoingDeviation
    TimeGapStrain
    LargeOutflowFlag
    PaymentTypeStrain
    SituationalStrainScore

NOTE:
  This is a *feature injector*, not a data-augmenter.
  It does NOT change labels or add synthetic transactions.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

PREFIX = "HI-Small"

BASE_DIR = Path(__file__).resolve().parent
TRANS_PATH = BASE_DIR / f"{PREFIX}_Trans.csv"
ACCOUNTS_PATH = BASE_DIR / f"{PREFIX}_accounts.csv"

OUT_DIR = BASE_DIR / "strain"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------


def load_data():
    print(f"[INFO] Loading transactions from {TRANS_PATH}")
    # We try to parse a 'Timestamp' column if present
    df_tx = pd.read_csv(TRANS_PATH)

    # If Timestamp exists, parse to datetime; otherwise set a synthetic time index
    if "Timestamp" in df_tx.columns:
        df_tx["Timestamp_dt"] = pd.to_datetime(
            df_tx["Timestamp"], errors="coerce", infer_datetime_format=True
        )
    else:
        # Synthetic timestamp: assume 1-minute spacing
        df_tx["Timestamp_dt"] = pd.to_datetime(
            pd.Series(range(len(df_tx))), unit="m"
        )

    print(f"[INFO] Loading accounts from {ACCOUNTS_PATH}")
    df_acc = pd.read_csv(ACCOUNTS_PATH)

    return df_tx, df_acc


# ---------------------------------------------------------------------
# ACCOUNT-LEVEL STRAIN FEATURES
# ---------------------------------------------------------------------


def compute_account_strain(df_tx: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-account strain feature table using the whole dataset
    (treated as approx. 90 days) and then scale to 7/30/90 day windows.
    """
    print("[INFO] Computing account-level strain features...")

    # These are the source and destination account columns
    src_col = "Account"
    dst_col = "Account.1" if "Account.1" in df_tx.columns else "To Account"

    # Ensure numeric amounts
    df_tx["Amount Paid"] = pd.to_numeric(df_tx["Amount Paid"], errors="coerce").fillna(0.0)
    df_tx["Amount Received"] = pd.to_numeric(df_tx["Amount Received"], errors="coerce").fillna(0.0)

    # Time span (for scaling windows)
    tmin = df_tx["Timestamp_dt"].min()
    tmax = df_tx["Timestamp_dt"].max()
    total_days = max((tmax - tmin).days, 1)

    # All accounts that appear either as src or dst
    all_accounts = pd.Index(
        sorted(set(df_tx[src_col]).union(set(df_tx[dst_col]))),
        name="Account_ID",
    )

    acc = pd.DataFrame(index=all_accounts)

    # In / out volumes + degrees
    out_grp = df_tx.groupby(src_col)
    in_grp = df_tx.groupby(dst_col)

    acc["TotalOut"] = out_grp["Amount Paid"].sum().reindex(all_accounts).fillna(0.0)
    acc["TotalIn"] = in_grp["Amount Received"].sum().reindex(all_accounts).fillna(0.0)
    acc["OutDegree"] = out_grp.size().reindex(all_accounts).fillna(0).astype(int)
    acc["InDegree"] = in_grp.size().reindex(all_accounts).fillna(0).astype(int)

    # Net flow over full period
    acc["NetFlow_full"] = acc["TotalIn"] - acc["TotalOut"]

    # Scale to 7/30/90-day equivalents
    days = float(total_days)
    acc["NetFlow_7d"] = acc["NetFlow_full"] * min(7.0 / days, 1.0)
    acc["NetFlow_30d"] = acc["NetFlow_full"] * min(30.0 / days, 1.0)
    acc["NetFlow_90d"] = acc["NetFlow_full"] * min(90.0 / days, 1.0)

    # Out/In ratio over full period -> treat as 30d metric
    eps = 1e-6
    acc["OutInRatio_30d"] = acc["TotalOut"] / (acc["TotalIn"] + eps)

    # Incoming volatility: std of received amounts
    incoming_std = in_grp["Amount Received"].std().reindex(all_accounts).fillna(0.0)
    acc["IncomingVolatility_30d"] = incoming_std

    # Liquidity shock count (approx 30d): outflow tx > mean + 2*std for that account
    out_mean = out_grp["Amount Paid"].mean().reindex(all_accounts).fillna(0.0)
    out_std = out_grp["Amount Paid"].std().reindex(all_accounts).fillna(0.0)

    def liquidity_shock_count(group):
        a = group["Amount Paid"]
        mu = a.mean()
        sd = a.std(ddof=0)
        thresh = mu + 2 * sd
        return (a > thresh).sum()

    shock_counts = out_grp.apply(liquidity_shock_count).reindex(all_accounts).fillna(0).astype(
        int
    )
    acc["LiquidityShockCount_30d"] = shock_counts

    # TopPartnerConcentration: outgoing amount concentration to top counterparty
    partner_amount = df_tx.groupby([src_col, dst_col])["Amount Paid"].sum()

    def top_partner_concentration(s):
        total = s.sum()
        if total <= 0:
            return 0.0
        return float(s.max() / total)

    tpc = partner_amount.groupby(level=0).apply(top_partner_concentration)
    acc["TopPartnerConcentration"] = tpc.reindex(all_accounts).fillna(0.0)

    # BankDependenceScore: outgoing amount concentration to top "To Bank"
    if "To Bank" in df_tx.columns:
        bank_amount = df_tx.groupby([src_col, "To Bank"])["Amount Paid"].sum()

        def bank_conc(s):
            total = s.sum()
            if total <= 0:
                return 0.0
            return float(s.max() / total)

        bdep = bank_amount.groupby(level=0).apply(bank_conc)
        acc["BankDependenceScore"] = bdep.reindex(all_accounts).fillna(0.0)
    else:
        acc["BankDependenceScore"] = 0.0

    # Network centrality: degree centrality approximation
    deg = acc["InDegree"] + acc["OutDegree"]
    max_deg = max(deg.max(), 1.0)
    acc["NetworkCentrality"] = deg / max_deg

    # -----------------------------------------------------------------
    # ChronicStrainScore: combine deficit, out/in ratio, volatility
    # -----------------------------------------------------------------
    # Economic deficit (only when out > in)
    deficit = (-acc["NetFlow_90d"]).clip(lower=0.0)  # positive when more out than in
    deficit_norm = deficit / (deficit.max() + eps)

    ratio = acc["OutInRatio_30d"].clip(lower=0.0)
    ratio_norm = ratio / (ratio.quantile(0.99) + eps)  # robust scaling

    vol = acc["IncomingVolatility_30d"].clip(lower=0.0)
    vol_norm = vol / (vol.quantile(0.99) + eps)

    acc["ChronicStrainScore"] = (deficit_norm + ratio_norm + vol_norm) / 3.0

    # -----------------------------------------------------------------
    # NetworkStrainExposure: mean neighbour ChronicStrainScore
    # -----------------------------------------------------------------
    print("[INFO] Computing NetworkStrainExposure...")

    neighbours = defaultdict(list)
    for src, dst in zip(df_tx[src_col], df_tx[dst_col]):
        neighbours[src].append(dst)
        neighbours[dst].append(src)

    exposure_vals = []
    for acc_id in acc.index:
        neigh_list = neighbours.get(acc_id, [])
        if not neigh_list:
            exposure_vals.append(0.0)
        else:
            exposure_vals.append(
                float(acc.loc[neigh_list, "ChronicStrainScore"].mean())
            )

    acc["NetworkStrainExposure"] = exposure_vals

    # Keep only requested columns with stable order
    acc_strain = acc[
        [
            "NetFlow_7d",
            "NetFlow_30d",
            "NetFlow_90d",
            "OutInRatio_30d",
            "IncomingVolatility_30d",
            "LiquidityShockCount_30d",
            "TopPartnerConcentration",
            "BankDependenceScore",
            "NetworkCentrality",
            "NetworkStrainExposure",
            "ChronicStrainScore",
        ]
    ].copy()
    acc_strain.index.name = "Account_ID"

    return acc_strain, out_mean, out_std


# ---------------------------------------------------------------------
# TRANSACTION-LEVEL STRAIN FEATURES
# ---------------------------------------------------------------------


def compute_transaction_strain(
    df_tx: pd.DataFrame,
    acc_strain: pd.DataFrame,
    out_mean: pd.Series,
    out_std: pd.Series,
) -> pd.DataFrame:
    """
    Add per-transaction strain features using account-level statistics.
    """
    print("[INFO] Computing transaction-level strain features...")

    src_col = "Account"
    dst_col = "Account.1" if "Account.1" in df_tx.columns else "To Account"

    # Make sure TxID exists
    df_tx = df_tx.copy()
    if "TxID" not in df_tx.columns:
        df_tx["TxID"] = np.arange(len(df_tx), dtype=np.int64)

    # Map per-account mean/std/p95 for outgoing amounts
    p95_out = (
        df_tx.groupby(src_col)["Amount Paid"]
        .quantile(0.95)
        .reindex(out_mean.index)
        .fillna(0.0)
    )

    df_tx["out_mean"] = df_tx[src_col].map(out_mean).fillna(0.0)
    df_tx["out_std"] = df_tx[src_col].map(out_std).fillna(0.0)
    df_tx["out_p95"] = df_tx[src_col].map(p95_out).fillna(0.0)

    # OutgoingDeviation: z-score of Amount Paid vs account's typical
    denom = df_tx["out_std"].replace(0.0, np.nan)
    dev = (df_tx["Amount Paid"] - df_tx["out_mean"]) / denom
    df_tx["OutgoingDeviation"] = dev.fillna(0.0)

    # TimeGapStrain: inverse of time gap to previous outgoing tx from same account
    df_tx = df_tx.sort_values(["Account", "Timestamp_dt", "TxID"])
    gap_hours = (
        df_tx.groupby(src_col)["Timestamp_dt"]
        .diff()
        .dt.total_seconds()
        .div(3600.0)
    )
    # Replace NaNs (first tx per account) with median gap
    median_gap = gap_hours.median()
    gap_hours = gap_hours.fillna(median_gap if pd.notna(median_gap) else 24.0)
    # Small gap -> high strain; use 1 / (1 + gap)
    df_tx["TimeGapStrain"] = 1.0 / (1.0 + gap_hours.clip(lower=0.0))

    # LargeOutflowFlag: Amount Paid >= 95th percentile for that account
    df_tx["LargeOutflowFlag"] = (
        df_tx["Amount Paid"] >= df_tx["out_p95"]
    ).astype(int)

    # PaymentTypeStrain: map payment format → 0..1 risk score
    risk_map = {
        "Reinvestment": 0.2,
        "ACH": 0.4,
        "Credit Card": 0.5,
        "Cheque": 0.7,
        "Wire": 0.8,
        "Cash": 1.0,
    }
    df_tx["PaymentTypeStrain"] = (
        df_tx.get("Payment Format", pd.Series(index=df_tx.index, dtype=str))
        .map(risk_map)
        .fillna(0.4)
    )

    # SituationalStrainScore: combine |OutgoingDeviation|, TimeGapStrain,
    # LargeOutflowFlag, PaymentTypeStrain into one 0..1 score
    dev_abs = df_tx["OutgoingDeviation"].abs()
    # squash with tanh so huge z-scores don't dominate
    dev_norm = np.tanh(dev_abs / 3.0)

    tgap = df_tx["TimeGapStrain"].clip(0.0, 1.0)
    ptype = df_tx["PaymentTypeStrain"].clip(0.0, 1.0)
    large_flag = df_tx["LargeOutflowFlag"].astype(float)

    # Weighted average; weights are somewhat heuristic
    situ = (dev_norm + tgap + ptype + 0.5 * large_flag) / 3.5
    df_tx["SituationalStrainScore"] = situ.clip(0.0, 1.0)

    # Drop helper columns
    df_tx = df_tx.drop(columns=["out_mean", "out_std", "out_p95"])

    return df_tx


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def main():
    df_tx, df_acc = load_data()

    # Account-level strain features
    acc_strain, out_mean, out_std = compute_account_strain(df_tx)

    # Merge strain features into accounts table
    acc_strain_reset = acc_strain.reset_index()
    df_acc_strain = df_acc.merge(
        acc_strain_reset,
        left_on="Account Number",
        right_on="Account_ID",
        how="left",
    )
    # If some accounts never appeared in transactions, fill with 0
    for col in acc_strain.columns:
        df_acc_strain[col] = df_acc_strain[col].fillna(0.0)

    # Transaction-level strain features
    df_tx_strain = compute_transaction_strain(df_tx, acc_strain, out_mean, out_std)

    # Save
    out_acc = OUT_DIR / f"{PREFIX}_accounts_STRAIN.csv"
    out_tx = OUT_DIR / f"{PREFIX}_Trans_STRAIN.csv"

    df_acc_strain.to_csv(out_acc, index=False)
    df_tx_strain.to_csv(out_tx, index=False)

    print("[INFO] --------------------------------------------------")
    print(f"[INFO] Saved accounts_with_strain to: {out_acc}")
    print(f"[INFO] Saved transactions_with_strain to: {out_tx}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
