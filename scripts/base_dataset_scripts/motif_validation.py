import pandas as pd
import matplotlib.pyplot as plt

tx = pd.read_csv(r"Datasets\baseline_RAT\transactions.csv")

# Check column presence
print("Columns:", list(tx.columns))

# Count injected motifs
print("\nInjected motif counts:")
print(tx['injected_motif'].value_counts(dropna=True))

# Check how many total injected edges
injected_count = tx['injected_motif'].notna().sum()
print(f"\nTotal injected motif edges: {injected_count}")

# Check some samples
print("\nSample injected rows:")
print(tx[tx['injected_motif'].notna()].head(10))

# Example: check a fan-in group (same bene_acct, multiple orig_acct)
fanin_group = tx[tx['injected_motif'] == 'fanin'].head(10)
target = fanin_group.iloc[0]['bene_acct']
print("Fan-in target:", target)
print(tx[(tx['bene_acct'] == target) & (tx['injected_motif'] == 'fanin')])

# Example: check a cycle (RAT_CYCLE)
cycle_df = tx[tx['alert_id'] == 'RAT_CYCLE'].head(4)
print("\nCycle example:\n", cycle_df[['orig_acct','bene_acct','tran_timestamp','alert_id','injected_motif']])

baseline = pd.read_csv(r"Datasets\baseline\transactions.csv")
rat = pd.read_csv(r"Datasets\baseline_RAT\transactions.csv")

print("Baseline tx count:", len(baseline))
print("RAT tx count:", len(rat))

plt.hist(baseline['base_amt'], bins=50, alpha=0.6, label='Baseline')
plt.hist(rat['base_amt'], bins=50, alpha=0.6, label='RAT-injected')
plt.legend(); plt.title("Transaction Amount Distribution"); plt.show()
