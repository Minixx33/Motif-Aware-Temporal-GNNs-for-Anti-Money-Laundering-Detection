import pandas as pd

low = pd.read_csv('HI-Small_Trans_RAT_low.csv')
med = pd.read_csv('HI-Small_Trans_RAT_medium.csv')
high = pd.read_csv('HI-Small_Trans_RAT_high.csv')

# Check if RAT features differ
rat_cols = [c for c in low.columns if 'RAT' in c]
print("Checking RAT features...")
for col in rat_cols:
    print(f"{col}:")
    print(f"  Low:  mean={low[col].mean():.4f}, std={low[col].std():.4f}")
    print(f"  Med:  mean={med[col].mean():.4f}, std={med[col].std():.4f}")
    print(f"  High: mean={high[col].mean():.4f}, std={high[col].std():.4f}")