import torch

# Check splits
print("=== Checking Splits ===")
train = torch.load('tgat_splits/HI-Small_Trans/train_edge_idx.pt')
val = torch.load('tgat_splits/HI-Small_Trans/val_edge_idx.pt')
test = torch.load('tgat_splits/HI-Small_Trans/test_edge_idx.pt')

print(f"Train: {len(train):,} edges")
print(f"Val:   {len(val):,} edges")
print(f"Test:  {len(test):,} edges")

# Check temporal ordering
timestamps = torch.load('tgat_graphs/HI-Small_Trans/timestamps.pt')

train_times = timestamps[train]
val_times = timestamps[val]
test_times = timestamps[test]

print(f"\n=== Time Ranges ===")
print(f"Train: {train_times.min()} → {train_times.max()}")
print(f"Val:   {val_times.min()} → {val_times.max()}")
print(f"Test:  {test_times.min()} → {test_times.max()}")

# Check for overlap
print(f"\n=== Temporal Check ===")
if train_times.max() <= val_times.min():
    print("✓ Train → Val: No overlap")
else:
    print("❌ Train → Val: OVERLAP DETECTED!")

if val_times.max() <= test_times.min():
    print("✓ Val → Test: No overlap")
else:
    print("❌ Val → Test: OVERLAP DETECTED!")

# Check labels
y_edge = torch.load('tgat_graphs/HI-Small_Trans/y_edge.pt')
print(f"\n=== Label Distribution ===")
print(f"Train fraud: {y_edge[train].sum()}/{len(train)} ({100*y_edge[train].float().mean():.2f}%)")
print(f"Val fraud:   {y_edge[val].sum()}/{len(val)} ({100*y_edge[val].float().mean():.2f}%)")
print(f"Test fraud:  {y_edge[test].sum()}/{len(test)} ({100*y_edge[test].float().mean():.2f}%)")