import pandas as pd

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
TX_PATH = r"C:\Users\yasmi\OneDrive\Desktop\Uni - Master's\Fall 2025\MLR 570\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection\ibm_transcations_datasets\HI-Small_Trans.csv"
ACCOUNTS_PATH = r"C:\Users\yasmi\OneDrive\Desktop\Uni - Master's\Fall 2025\MLR 570\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection\ibm_transcations_datasets\HI-Small_accounts.csv"

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
tx = pd.read_csv(TX_PATH)
acc = pd.read_csv(ACCOUNTS_PATH)

# ------------------------------------------------------------
# 1. Number of transactions (edges)
# ------------------------------------------------------------
num_edges = len(tx)

# ------------------------------------------------------------
# 2. Number of accounts (nodes)
# ------------------------------------------------------------
unique_accounts = pd.unique(pd.concat([tx["Account"], tx["Account.1"]]))
num_nodes = len(unique_accounts)

# ------------------------------------------------------------
# 3. Number of banks
# ------------------------------------------------------------
bank_col = "Bank ID" if "Bank ID" in acc.columns else "Bank Name"
num_banks = acc[bank_col].nunique()

# ------------------------------------------------------------
# 4. Compute % laundering edges
# ------------------------------------------------------------
label_col = "Label" if "Label" in tx.columns else "Is Laundering"
pct_laundering = tx[label_col].mean() * 100

# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------
print("\n=== HI-Small_Trans Dataset Summary ===\n")
print(f"# transactions (edges): {num_edges:,}")
print(f"# accounts (nodes):     {num_nodes:,}")
print(f"# banks:                {num_banks}")
print(f"% laundering edges:     {pct_laundering:.3f}%") 
