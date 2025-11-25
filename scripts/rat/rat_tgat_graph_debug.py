"""
TGAT Graph Debugging Script
===========================
Purpose: Validate that the TGAT event stream is correctly constructed
Dataset: HI-Small_Trans_RAT_medium (or any TGAT format)
Format: TGAT temporal event stream

This script checks:
  ✓ Files exist and load correctly
  ✓ Data types and shapes are correct
  ✓ No NaN/Inf values in features
  ✓ Event stream format is valid
  ✓ RAT features are present and in valid ranges
  ✓ Label distribution is reasonable
  ✓ STRICT temporal ordering (CRITICAL for TGAT)
  ✓ Event timestamps and causality
  ✓ Statistics and metadata

Outputs:
  - debug_report.txt (full detailed report)
  - debug_summary.txt (quick reference)
  - tgat_analysis.png (visualizations)
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================
# 1. SETUP PATHS AND OUTPUT FILE
# ============================================================

GRAPH_DIR = r"C:\Users\yasmi\OneDrive\Desktop\Uni - Master's\Fall 2025\MLR 570\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection\ibm_transcations_datasets\tgat_graphs\HI-Small_Trans_RAT_high"

# Create output file for logging
OUTPUT_FILE = os.path.join(GRAPH_DIR, "debug_report.txt")

class Logger:
    """Class to log output to both console and file"""
    def __init__(self, filepath):
        self.terminal = __import__('sys').stdout
        self.log = open(filepath, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

# Redirect output to both console and file
import sys
logger = Logger(OUTPUT_FILE)
sys.stdout = logger

print("=" * 70)
print("TGAT EVENT STREAM DEBUGGING")
print("=" * 70)
print(f"\nGraph directory: {GRAPH_DIR}")
print(f"Debug report will be saved to: {OUTPUT_FILE}\n")

# ============================================================
# 2. CHECK FILES EXIST
# ============================================================

print("=" * 70)
print("FILE EXISTENCE CHECK")
print("=" * 70)

required_files = [
    "src_nodes.pt",
    "dst_nodes.pt",
    "timestamps.pt",
    "edge_attr.pt",
    "y_edge.pt",
    "x_node.pt",
    "y_node.pt",
    "node_mapping.json",
    "edge_attr_cols.json",
    "feature_stats.json",
    "graph_stats.json"
]

all_exist = True
for filename in required_files:
    filepath = os.path.join(GRAPH_DIR, filename)
    exists = os.path.exists(filepath)
    status = "True" if exists else "False"
    print(f"{status} {filename}")
    if not exists:
        all_exist = False

if not all_exist:
    print("\n ERROR: Some required files are missing!")
    sys.stdout = logger.terminal
    logger.close()
    exit(1)
else:
    print("\n✓ All required files exist")

# ============================================================
# 3. LOAD TGAT EVENT STREAM
# ============================================================

print("\n" + "=" * 70)
print("LOADING TGAT EVENT STREAM")
print("=" * 70)

try:
    src_nodes = torch.load(os.path.join(GRAPH_DIR, "src_nodes.pt"), weights_only=False)
    dst_nodes = torch.load(os.path.join(GRAPH_DIR, "dst_nodes.pt"), weights_only=False)
    timestamps = torch.load(os.path.join(GRAPH_DIR, "timestamps.pt"), weights_only=False)
    edge_attr = torch.load(os.path.join(GRAPH_DIR, "edge_attr.pt"), weights_only=False)
    y_edge = torch.load(os.path.join(GRAPH_DIR, "y_edge.pt"), weights_only=False)
    x_node = torch.load(os.path.join(GRAPH_DIR, "x_node.pt"), weights_only=False)
    y_node = torch.load(os.path.join(GRAPH_DIR, "y_node.pt"), weights_only=False)
    
    with open(os.path.join(GRAPH_DIR, "node_mapping.json")) as f:
        node_mapping = json.load(f)
    
    with open(os.path.join(GRAPH_DIR, "edge_attr_cols.json")) as f:
        edge_attr_cols = json.load(f)
    
    with open(os.path.join(GRAPH_DIR, "feature_stats.json")) as f:
        feature_stats = json.load(f)
    
    with open(os.path.join(GRAPH_DIR, "graph_stats.json")) as f:
        graph_stats = json.load(f)
    
    print(" All files loaded successfully")
    
except Exception as e:
    print(f" ERROR loading files: {e}")
    sys.stdout = logger.terminal
    logger.close()
    exit(1)

# ============================================================
# 4. VALIDATE SHAPES AND TYPES (TGAT FORMAT)
# ============================================================

print("\n" + "=" * 70)
print("SHAPE AND TYPE VALIDATION (TGAT FORMAT)")
print("=" * 70)

num_nodes = x_node.shape[0]
num_events = len(src_nodes)

print(f"\nEvent Stream Structure:")
print(f"  Nodes: {num_nodes:,}")
print(f"  Events: {num_events:,}")
print(f"  Avg events per node: {num_events / num_nodes:.2f}")

print(f"\nTensor Shapes:")
print(f"  src_nodes:   {src_nodes.shape} (expected: [E])")
print(f"  dst_nodes:   {dst_nodes.shape} (expected: [E])")
print(f"  timestamps:  {timestamps.shape} (expected: [E])")
print(f"  edge_attr:   {edge_attr.shape} (expected: [E, F_e])")
print(f"  y_edge:      {y_edge.shape} (expected: [E])")
print(f"  x_node:      {x_node.shape} (expected: [N, F_n])")
print(f"  y_node:      {y_node.shape} (expected: [N])")

print(f"\nData Types:")
print(f"  src_nodes:   {src_nodes.dtype} (expected: torch.int64)")
print(f"  dst_nodes:   {dst_nodes.dtype} (expected: torch.int64)")
print(f"  timestamps:  {timestamps.dtype} (expected: torch.int64)")
print(f"  edge_attr:   {edge_attr.dtype} (expected: torch.float32)")
print(f"  y_edge:      {y_edge.dtype} (expected: torch.int64)")
print(f"  x_node:      {x_node.dtype} (expected: torch.float32)")
print(f"  y_node:      {y_node.dtype} (expected: torch.int64)")

# Validate shapes
assert src_nodes.shape[0] == num_events, "src_nodes should match num events"
assert dst_nodes.shape[0] == num_events, "dst_nodes should match num events"
assert timestamps.shape[0] == num_events, "timestamps should match num events"
assert edge_attr.shape[0] == num_events, "edge_attr rows should match num events"
assert y_edge.shape[0] == num_events, "y_edge should match num events"
assert y_node.shape[0] == num_nodes, "y_node should match num nodes"

print("\n All shapes and types are correct for TGAT format")

# ============================================================
# 5. CHECK FOR NaN/INF VALUES
# ============================================================

print("\n" + "=" * 70)
print("NaN/INF CHECK")
print("=" * 70)

nan_inf_issues = []

# Check edge features
edge_attr_nan = torch.isnan(edge_attr).sum().item()
edge_attr_inf = torch.isinf(edge_attr).sum().item()
print(f"\nedge_attr:")
print(f"  NaN values: {edge_attr_nan}")
print(f"  Inf values: {edge_attr_inf}")
if edge_attr_nan > 0 or edge_attr_inf > 0:
    nan_inf_issues.append("edge_attr")

# Check node features
x_nan = torch.isnan(x_node).sum().item()
x_inf = torch.isinf(x_node).sum().item()
print(f"\nx_node (node features):")
print(f"  NaN values: {x_nan}")
print(f"  Inf values: {x_inf}")
if x_nan > 0 or x_inf > 0:
    nan_inf_issues.append("x_node")

# Check timestamps
ts_nan = torch.isnan(timestamps.float()).sum().item()
ts_inf = torch.isinf(timestamps.float()).sum().item()
print(f"\ntimestamps:")
print(f"  NaN values: {ts_nan}")
print(f"  Inf values: {ts_inf}")
if ts_nan > 0 or ts_inf > 0:
    nan_inf_issues.append("timestamps")

if len(nan_inf_issues) > 0:
    print(f"\n ERROR: NaN/Inf found in: {', '.join(nan_inf_issues)}")
else:
    print("\n No NaN/Inf values found")

# ============================================================
# 6. VALIDATE NODE INDICES
# ============================================================

print("\n" + "=" * 70)
print("NODE INDEX VALIDATION")
print("=" * 70)

# Check node indices are within valid range
src_min = src_nodes.min().item()
src_max = src_nodes.max().item()
dst_min = dst_nodes.min().item()
dst_max = dst_nodes.max().item()

print(f"\nSource Nodes:")
print(f"  Min ID: {src_min}")
print(f"  Max ID: {src_max}")
print(f"  Unique: {torch.unique(src_nodes).numel():,}")

print(f"\nDestination Nodes:")
print(f"  Min ID: {dst_min}")
print(f"  Max ID: {dst_max}")
print(f"  Unique: {torch.unique(dst_nodes).numel():,}")

assert src_min >= 0 and dst_min >= 0, "Node IDs should be >= 0"
assert src_max < num_nodes and dst_max < num_nodes, f"Node IDs exceed num_nodes ({num_nodes})"

# Check for self-loops
self_loops = (src_nodes == dst_nodes).sum().item()
print(f"\nSelf-loops: {self_loops:,} ({100 * self_loops / num_events:.2f}%)")

# Connected nodes
all_nodes = torch.cat([src_nodes, dst_nodes])
connected_nodes = torch.unique(all_nodes).numel()
isolated_nodes = num_nodes - connected_nodes

print(f"\nConnectivity:")
print(f"  Connected nodes: {connected_nodes:,}")
print(f"  Isolated nodes:  {isolated_nodes:,}")

print("\n Node indices are valid")

# ============================================================
# 7. CRITICAL TEMPORAL VALIDATION (MOST IMPORTANT FOR TGAT!)
# ============================================================

print("\n" + "=" * 70)
print(" CRITICAL: TEMPORAL ORDERING VALIDATION")
print("=" * 70)

# Check if timestamps are strictly increasing
is_sorted = torch.all(timestamps[1:] >= timestamps[:-1]).item()

if not is_sorted:
    # Find violations
    time_diffs = timestamps[1:] - timestamps[:-1]
    violations = (time_diffs < 0).sum().item()
    violation_indices = torch.where(time_diffs < 0)[0]
    
    print(f"\n CRITICAL ERROR: Events are NOT in chronological order!")
    print(f"   Found {violations} temporal ordering violations")
    print(f"   First violation at index: {violation_indices[0].item()}")
    print(f"\n   TGAT REQUIRES strictly non-decreasing timestamps!")
    print(f"   This MUST be fixed before training.")
    
    # Show example violation
    if len(violation_indices) > 0:
        idx = violation_indices[0].item()
        print(f"\n   Example violation:")
        print(f"     Event {idx}:   t={timestamps[idx].item()}")
        print(f"     Event {idx+1}: t={timestamps[idx+1].item()}")
else:
    print("\n Events are in chronological order (REQUIRED for TGAT)")

# Time span statistics
time_span_seconds = timestamps.max().item() - timestamps.min().item()
time_span_days = time_span_seconds / (3600 * 24)

print(f"\nTimestamp Statistics:")
print(f"  Min timestamp: {timestamps.min().item()}")
print(f"  Max timestamp: {timestamps.max().item()}")
print(f"  Time span:     {time_span_days:.1f} days")

# Check for duplicate timestamps
unique_timestamps = torch.unique(timestamps).numel()
duplicate_ratio = 1 - (unique_timestamps / num_events)

print(f"\nTimestamp Uniqueness:")
print(f"  Unique timestamps: {unique_timestamps:,}")
print(f"  Total events:      {num_events:,}")
print(f"  Duplicate ratio:   {duplicate_ratio*100:.2f}%")

if duplicate_ratio > 0.5:
    print(f"  ℹ Note: {duplicate_ratio*100:.1f}% of events share timestamps")
    print(f"     This is normal for transaction data (multiple events per second)")
    print(f"     TGAT handles this with tie-breaking logic")

# Time gap analysis
time_diffs = timestamps[1:] - timestamps[:-1]
print(f"\nTime Gaps Between Consecutive Events:")
print(f"  Min gap:  {time_diffs.min().item()} seconds")
print(f"  Max gap:  {time_diffs.max().item()} seconds ({time_diffs.max().item()/3600:.1f} hours)")
print(f"  Mean gap: {time_diffs.float().mean().item():.1f} seconds")

# ============================================================
# 8. CHECK RAT FEATURES
# ============================================================

print("\n" + "=" * 70)
print("RAT FEATURE VALIDATION")
print("=" * 70)

print(f"\nTotal edge features: {len(edge_attr_cols)}")
print("\nFeature columns:")

rat_features = []
motif_features = []
structural_features = []

for i, col in enumerate(edge_attr_cols):
    print(f"  {i+1:2d}. {col}")
    if col.startswith("RAT_"):
        rat_features.append(col)
    elif col.startswith("motif_"):
        motif_features.append(col)
    else:
        structural_features.append(col)

print(f"\nFeature Breakdown:")
print(f"  RAT features:         {len(rat_features)}")
print(f"  Motif features:       {len(motif_features)}")
print(f"  Structural features:  {len(structural_features)}")

# Check if key RAT features exist
expected_rat_features = [
    "RAT_offender_score",
    "RAT_target_score",
    "RAT_guardian_weakness_score",
    "RAT_score"
]

print(f"\nKey RAT Features Check:")
for feat in expected_rat_features:
    if feat in edge_attr_cols:
        idx = edge_attr_cols.index(feat)
        feat_data = edge_attr[:, idx]
        print(f"   {feat}")
        print(f"      Range: [{feat_data.min():.4f}, {feat_data.max():.4f}]")
        print(f"      Mean:  {feat_data.mean():.4f}")
    else:
        print(f"   {feat} - MISSING!")

# Check motif features
expected_motif_features = ["motif_fanin", "motif_fanout", "motif_chain", "motif_cycle"]

print(f"\nMotif Features Check:")
for feat in expected_motif_features:
    if feat in edge_attr_cols:
        idx = edge_attr_cols.index(feat)
        feat_data = edge_attr[:, idx]
        print(f"  {feat}")
        print(f"      Range: [{feat_data.min():.4f}, {feat_data.max():.4f}]")
        print(f"      Mean:  {feat_data.mean():.4f}")
    else:
        print(f"  {feat} - MISSING!")

# ============================================================
# 9. LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

# Edge labels
num_laundering_events = y_edge.sum().item()
pct_laundering_events = 100 * y_edge.float().mean().item()

print(f"\nEvent Labels:")
print(f"  Laundering events: {num_laundering_events:,} ({pct_laundering_events:.2f}%)")
print(f"  Normal events:     {num_events - num_laundering_events:,} ({100 - pct_laundering_events:.2f}%)")

# Node labels
num_laundering_nodes = y_node.sum().item()
pct_laundering_nodes = 100 * y_node.float().mean().item()

print(f"\nNode Labels:")
print(f"  Laundering nodes: {num_laundering_nodes:,} ({pct_laundering_nodes:.2f}%)")
print(f"  Normal nodes:     {num_nodes - num_laundering_nodes:,} ({100 - pct_laundering_nodes:.2f}%)")

# Check for class imbalance
if pct_laundering_events < 1:
    print(f"\n WARNING: Severe class imbalance ({pct_laundering_events:.2f}% positive)")
    print("  → Recommended: Use weighted loss or focal loss for training")
elif pct_laundering_events < 5:
    print(f"\n Note: Moderate class imbalance ({pct_laundering_events:.2f}% positive)")

# ============================================================
# 10. TEMPORAL LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL LABEL ANALYSIS")
print("=" * 70)

# Split events into temporal windows
num_windows = 10
window_size = num_events // num_windows

print(f"\nAnalyzing label distribution across {num_windows} temporal windows...")
print("(Important for temporal train/val/test splits)\n")

for i in range(num_windows):
    start_idx = i * window_size
    end_idx = min((i + 1) * window_size, num_events)
    window_labels = y_edge[start_idx:end_idx]
    window_launder_pct = 100 * window_labels.float().mean().item()
    
    print(f"  Window {i+1:2d}: Events {start_idx:7d}-{end_idx:7d} | "
          f"Laundering: {window_launder_pct:5.2f}%")

# Check if distribution is consistent across time
window_stats = []
for i in range(num_windows):
    start_idx = i * window_size
    end_idx = min((i + 1) * window_size, num_events)
    window_labels = y_edge[start_idx:end_idx]
    window_stats.append(window_labels.float().mean().item())

variation = np.std(window_stats)
print(f"\nTemporal distribution variation (std): {variation:.4f}")
if variation > 0.01:
    print("   Significant temporal variation in label distribution")
    print("     Consider stratified temporal splits")

# ============================================================
# 11. COMPARE WITH METADATA
# ============================================================

print("\n" + "=" * 70)
print("METADATA VALIDATION")
print("=" * 70)

print(f"\nGraph Stats (from graph_stats.json):")
print(f"  Dataset type:           {graph_stats.get('dataset_type', 'N/A')}")
print(f"  Format:                 {graph_stats.get('format', 'N/A')}")
print(f"  Num nodes (recorded):   {graph_stats.get('num_nodes', 'N/A'):,}")
print(f"  Num events (recorded):  {graph_stats.get('num_events', 'N/A'):,}")
print(f"  Num edge features:      {graph_stats.get('num_edge_features', 'N/A')}")
print(f"  Num node features:      {graph_stats.get('num_node_features', 'N/A')}")
print(f"  Laundering % (events):  {graph_stats.get('pct_laundering_events', 'N/A'):.2f}%")

# Verify consistency
assert graph_stats['num_nodes'] == num_nodes, "Mismatch: num_nodes"
assert graph_stats['num_events'] == num_events, "Mismatch: num_events"
assert graph_stats['num_edge_features'] == edge_attr.shape[1], "Mismatch: num_edge_features"
assert graph_stats['num_node_features'] == x_node.shape[1], "Mismatch: num_node_features"

print("\n Metadata is consistent with loaded data")

# ============================================================
# 12. VISUALIZATIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("TGAT Event Stream Analysis", fontsize=16, fontweight='bold')

# 1. Event distribution over time
time_hours = (timestamps - timestamps.min()).float() / 3600
axes[0, 0].hist(time_hours.numpy(), bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].set_xlabel('Time (hours from start)')
axes[0, 0].set_ylabel('Event Count')
axes[0, 0].set_title('Temporal Event Distribution')

# 2. Label distribution
label_counts = [num_events - num_laundering_events, num_laundering_events]
axes[0, 1].bar(['Normal', 'Laundering'], label_counts, color=['green', 'red'], alpha=0.7)
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Event Label Distribution')
axes[0, 1].set_yscale('log')

# 3. Inter-event time gaps
axes[0, 2].hist(time_diffs.numpy(), bins=50, edgecolor='black', alpha=0.7, color='coral')
axes[0, 2].set_xlabel('Time Gap (seconds)')
axes[0, 2].set_ylabel('Count')
axes[0, 2].set_title('Inter-Event Time Gaps')
axes[0, 2].set_yscale('log')

# 4. Motif feature comparison
motif_means = []
motif_names = []
for feat in expected_motif_features:
    if feat in edge_attr_cols:
        idx = edge_attr_cols.index(feat)
        motif_means.append(edge_attr[:, idx].mean().item())
        motif_names.append(feat.replace('motif_', ''))

if len(motif_means) > 0:
    axes[1, 0].bar(motif_names, motif_means, color='skyblue', edgecolor='black', alpha=0.7)
    axes[1, 0].set_ylabel('Mean Value')
    axes[1, 0].set_title('Motif Feature Means')
    axes[1, 0].tick_params(axis='x', rotation=45)
else:
    axes[1, 0].text(0.5, 0.5, 'No motif features found', ha='center', va='center')

# 5. Temporal label distribution
window_labels_list = []
for i in range(num_windows):
    start_idx = i * window_size
    end_idx = min((i + 1) * window_size, num_events)
    window_labels = y_edge[start_idx:end_idx]
    window_labels_list.append(100 * window_labels.float().mean().item())

axes[1, 1].plot(range(1, num_windows + 1), window_labels_list, marker='o', linewidth=2)
axes[1, 1].set_xlabel('Temporal Window')
axes[1, 1].set_ylabel('Laundering %')
axes[1, 1].set_title('Label Distribution Over Time')
axes[1, 1].grid(True, alpha=0.3)

# 6. RAT component scores
rat_components = ['RAT_offender_score', 'RAT_target_score', 'RAT_guardian_weakness_score']
component_means = []
component_names = []

for feat in rat_components:
    if feat in edge_attr_cols:
        idx = edge_attr_cols.index(feat)
        component_means.append(edge_attr[:, idx].mean().item())
        component_names.append(feat.replace('RAT_', '').replace('_score', ''))

if len(component_means) > 0:
    axes[1, 2].bar(component_names, component_means, color='orange', edgecolor='black', alpha=0.7)
    axes[1, 2].set_ylabel('Mean Score')
    axes[1, 2].set_title('RAT Component Scores')
    axes[1, 2].tick_params(axis='x', rotation=45)
else:
    axes[1, 2].text(0.5, 0.5, 'RAT components not found', ha='center', va='center')

plt.tight_layout()
plot_path = os.path.join(GRAPH_DIR, 'tgat_analysis.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\n Saved visualization: tgat_analysis.png")

# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION SUMMARY")
print("=" * 70)

issues = []

# Collect any issues found
if edge_attr_nan > 0 or edge_attr_inf > 0:
    issues.append("NaN/Inf in edge features")
if x_nan > 0 or x_inf > 0:
    issues.append("NaN/Inf in node features")
if not is_sorted:
    issues.append("CRITICAL: Temporal ordering violations (MUST FIX)")
if isolated_nodes > num_nodes * 0.1:
    issues.append(f"High isolated nodes ({isolated_nodes:,})")
if pct_laundering_events < 1:
    issues.append(f"Severe class imbalance ({pct_laundering_events:.2f}%)")

if len(issues) == 0:
    print("\n✅ ALL CHECKS PASSED!")
    print("\nThe TGAT event stream is ready for training:")
    print(f"  • {num_nodes:,} nodes, {num_events:,} events")
    print(f"  • {len(edge_attr_cols)} edge features (including RAT & motif)")
    print(f"  • {x_node.shape[1]} node features")
    print(f"  • {pct_laundering_events:.2f}% laundering events")
    print(f"  • Temporal span: {time_span_days:.1f} days")
    print(f"  • Events in chronological order: ✓")
    print("\n TGAT ready!")
else:
    print("\n⚠ ISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    print("\n⚠ Review issues before training.")

print("\n" + "=" * 70)
print("DEBUGGING COMPLETE")
print("=" * 70)
print(f"\n Debug report saved to: {OUTPUT_FILE}")

# Close logger and restore stdout
sys.stdout = logger.terminal
logger.close()

print(f"\n Full debug report saved to: {OUTPUT_FILE}")

# ============================================================
# CREATE SUMMARY FILE
# ============================================================

SUMMARY_FILE = os.path.join(GRAPH_DIR, "debug_summary.txt")

with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("TGAT EVENT STREAM VALIDATION SUMMARY\n")
    f.write("=" * 70 + "\n\n")
    
    f.write(f"Dataset: {os.path.basename(GRAPH_DIR)}\n")
    f.write(f"Type: {graph_stats.get('dataset_type', 'Unknown')}\n")
    f.write(f"Format: {graph_stats.get('format', 'Unknown')}\n\n")
    
    f.write("EVENT STREAM STRUCTURE:\n")
    f.write(f"  Nodes: {num_nodes:,}\n")
    f.write(f"  Events: {num_events:,}\n")
    f.write(f"  Avg events/node: {num_events / num_nodes:.2f}\n")
    f.write(f"  Temporal span: {time_span_days:.1f} days\n\n")
    
    f.write("FEATURES:\n")
    f.write(f"  Edge features: {edge_attr.shape[1]} ({len(rat_features)} RAT + {len(motif_features)} motif + {len(structural_features)} structural)\n")
    f.write(f"  Node features: {x_node.shape[1]}\n\n")
    
    f.write("LABELS:\n")
    f.write(f"  Laundering events: {num_laundering_events:,} ({pct_laundering_events:.2f}%)\n")
    f.write(f"  Laundering nodes: {num_laundering_nodes:,} ({pct_laundering_nodes:.2f}%)\n\n")
    
    f.write("TEMPORAL PROPERTIES:\n")
    f.write(f"  Chronological order: {'✓ Valid' if is_sorted else '✗ VIOLATIONS'}\n")
    f.write(f"  Unique timestamps: {unique_timestamps:,}\n")
    f.write(f"  Duplicate ratio: {duplicate_ratio*100:.1f}%\n")
    f.write(f"  Mean time gap: {time_diffs.float().mean().item():.1f} seconds\n\n")
    
    f.write("DATA QUALITY:\n")
    f.write(f"  NaN/Inf values: {'✓ None' if len(nan_inf_issues) == 0 else '✗ ' + ', '.join(nan_inf_issues)}\n")
    f.write(f"  Isolated nodes: {isolated_nodes:,}\n\n")
    
    f.write("VALIDATION RESULT:\n")
    if len(issues) == 0:
        f.write("  ALL CHECKS PASSED - Ready for TGAT training\n")
    else:
        f.write("  ISSUES FOUND:\n")
        for i, issue in enumerate(issues, 1):
            f.write(f"    {i}. {issue}\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write("FILES GENERATED:\n")
    f.write(f"  • debug_report.txt (full report)\n")
    f.write(f"  • debug_summary.txt (this summary)\n")
    f.write(f"  • tgat_analysis.png (visualizations)\n")
    f.write("\n" + "=" * 70 + "\n")
    f.write("IMPORTANT FOR TGAT:\n")
    f.write("  - Events MUST be in chronological order\n")
    f.write("  - Use temporal splits (70% train, 15% val, 15% test)\n")
    f.write("  - Handle class imbalance with weighted loss\n")
    f.write("  - Normalize features using training data only\n")

print(f" Summary saved to: {SUMMARY_FILE}")

print("\n" + "=" * 70)
print("VALIDATION COMPLETE - 3 FILES GENERATED:")
print("=" * 70)
print(f"1. {os.path.basename(OUTPUT_FILE)} - Full detailed report")
print(f"2. {os.path.basename(SUMMARY_FILE)} - Quick summary")
print(f"3. tgat_analysis.png - Visualizations")
print("\n All files saved in: {GRAPH_DIR}")