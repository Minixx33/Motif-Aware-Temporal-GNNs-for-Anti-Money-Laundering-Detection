import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

tx = pd.read_csv(r"Datasets/baseline_RAT/transactions.csv", low_memory=False)
feat = pd.read_csv(r"Datasets/baseline_RAT/motif_features.csv")

# mark account as suspicious if it appears in any SAR transaction
sar_accounts = set(tx.loc[tx["is_sar"]==1, "orig_acct"]) | set(tx.loc[tx["is_sar"]==1, "bene_acct"])
feat["is_sar_account"] = feat["acct_id"].isin(sar_accounts).astype(int)

feat.to_csv(r"Datasets/baseline_RAT/motif_features_labeled.csv", index=False)

