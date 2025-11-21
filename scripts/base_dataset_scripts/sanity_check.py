import pandas as pd
import matplotlib.pyplot as plt

feat = pd.read_csv(r"Datasets\baseline_RAT\motif_features.csv")

# Find rows where at least one motif role column > 0
motif_cols = [c for c in feat.columns if "_as_" in c]
motif_accounts = feat[feat[motif_cols].sum(axis=1) > 0]

print("Accounts with motifs:", len(motif_accounts))
print(motif_accounts.head())

print(feat[motif_cols].sum().sort_values(ascending=False))


plt.hist(feat["motif_participation"], bins=50)
plt.title("Distribution of motif participation per account")
plt.xlabel("Number of motif edges")
plt.ylabel("Count of accounts")
plt.show()
