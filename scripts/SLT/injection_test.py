import pandas as pd

path = r"C:\Users\kenzi\Documents\GitHub\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection\ibm_transcations_datasets\base_dataset_small\SLT\HI-Small_Trans_SLT_high.csv"

df = pd.read_csv(path)

launder = df[df["Is Laundering"] == 1][["SLT_score", "SLT_injected"]].copy()

print("Laundering rows:", len(launder))
print("Injected rows:", int(launder["SLT_injected"].sum()))
print(launder["SLT_score"].describe())
print("Unique scores:", sorted(launder["SLT_score"].unique()))