"""
RAT Graph Debugging Notebook
============================
Purpose: Validate that the RAT medium dataset's constructed graph is correct
Dataset: HI-Small_Trans_RAT_medium
Format: PyTorch Geometric (for GraphSAGE/GraphSAGE-T)

This notebook checks:
  ✓ Files exist and load correctly
  ✓ Data types and shapes are correct
  ✓ No NaN/Inf values in features
  ✓ Graph connectivity is valid
  ✓ RAT features are present and in valid ranges
  ✓ Label distribution is reasonable
  ✓ Temporal ordering (for GraphSAGE-T)
  ✓ Statistics and metadata
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
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fix OpenMP warning

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================
# 1. SETUP PATHS AND OUTPUT FILE
# ============================================================

GRAPH_DIR = r"C:\Users\yasmi\OneDrive\Desktop\Uni - Master's\Fall 2025\MLR 570\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection\ibm_transcations_datasets\graphs\HI-Small_Trans_RAT_high"

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
print("RAT MEDIUM GRAPH DEBUGGING")
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
    "edge_index.pt",
    "edge_attr.pt",
    "x.pt",
    "timestamps.pt",
    "y_edge.pt",
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
    status = "✓" if exists else "✗"
    print(f"{status} {filename}")
    if not exists:
        all_exist = False

if not all_exist:
    print("\n❌ ERROR: Some required files are missing!")
    exit(1)
else:
    print("\n✓ All required files exist")

# ============================================================
# 3. LOAD GRAPH DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING GRAPH DATA")
print("=" * 70)

try:
    edge_index = torch.load(os.path.join(GRAPH_DIR, "edge_index.pt"), weights_only=False)
    edge_attr = torch.load(os.path.join(GRAPH_DIR, "edge_attr.pt"), weights_only=False)
    x = torch.load(os.path.join(GRAPH_DIR, "x.pt"), weights_only=False)
    timestamps = torch.load(os.path.join(GRAPH_DIR, "timestamps.pt"), weights_only=False)
    y_edge = torch.load(os.path.join(GRAPH_DIR, "y_edge.pt"), weights_only=False)
    y_node = torch.load(os.path.join(GRAPH_DIR, "y_node.pt"), weights_only=False)
    
    with open(os.path.join(GRAPH_DIR, "node_mapping.json")) as f:
        node_mapping = json.load(f)
    
    with open(os.path.join(GRAPH_DIR, "edge_attr_cols.json")) as f:
        edge_attr_cols = json.load(f)
    
    with open(os.path.join(GRAPH_DIR, "feature_stats.json")) as f:
        feature_stats = json.load(f)
    
    with open(os.path.join(GRAPH_DIR, "graph_stats.json")) as f:
        graph_stats = json.load(f)
    
    print("✓ All files loaded successfully")
    
except Exception as e:
    print(f"❌ ERROR loading files: {e}")
    exit(1)

# ============================================================
# 4. VALIDATE SHAPES AND TYPES
# ============================================================

print("\n" + "=" * 70)
print("SHAPE AND TYPE VALIDATION")
print("=" * 70)

num_nodes = x.shape[0]
num_edges = edge_index.shape[1]

print(f"\nGraph Structure:")
print(f"  Nodes: {num_nodes:,}")
print(f"  Edges: {num_edges:,}")
print(f"  Avg degree: {num_edges / num_nodes:.2f}")

print(f"\nTensor Shapes:")
print(f"  edge_index:  {edge_index.shape} (expected: [2, E])")
print(f"  edge_attr:   {edge_attr.shape} (expected: [E, F_e])")
print(f"  x:           {x.shape} (expected: [N, F_n])")
print(f"  timestamps:  {timestamps.shape} (expected: [E])")
print(f"  y_edge:      {y_edge.shape} (expected: [E])")
print(f"  y_node:      {y_node.shape} (expected: [N])")

print(f"\nData Types:")
print(f"  edge_index:  {edge_index.dtype} (expected: torch.int64)")
print(f"  edge_attr:   {edge_attr.dtype} (expected: torch.float32)")
print(f"  x:           {x.dtype} (expected: torch.float32)")
print(f"  timestamps:  {timestamps.dtype} (expected: torch.int64)")
print(f"  y_edge:      {y_edge.dtype} (expected: torch.int64)")
print(f"  y_node:      {y_node.dtype} (expected: torch.int64)")

# Validate shapes
assert edge_index.shape[0] == 2, "edge_index should have shape [2, E]"
assert edge_attr.shape[0] == num_edges, "edge_attr rows should match num edges"
assert timestamps.shape[0] == num_edges, "timestamps should match num edges"
assert y_edge.shape[0] == num_edges, "y_edge should match num edges"
assert y_node.shape[0] == num_nodes, "y_node should match num nodes"

print("\n✓ All shapes and types are correct")

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
x_nan = torch.isnan(x).sum().item()
x_inf = torch.isinf(x).sum().item()
print(f"\nx (node features):")
print(f"  NaN values: {x_nan}")
print(f"  Inf values: {x_inf}")
if x_nan > 0 or x_inf > 0:
    nan_inf_issues.append("x")

# Check timestamps
ts_nan = torch.isnan(timestamps.float()).sum().item()
ts_inf = torch.isinf(timestamps.float()).sum().item()
print(f"\ntimestamps:")
print(f"  NaN values: {ts_nan}")
print(f"  Inf values: {ts_inf}")
if ts_nan > 0 or ts_inf > 0:
    nan_inf_issues.append("timestamps")

if len(nan_inf_issues) > 0:
    print(f"\n❌ ERROR: NaN/Inf found in: {', '.join(nan_inf_issues)}")
else:
    print("\n✓ No NaN/Inf values found")

# ============================================================
# 6. VALIDATE GRAPH CONNECTIVITY
# ============================================================

print("\n" + "=" * 70)
print("GRAPH CONNECTIVITY VALIDATION")
print("=" * 70)

# Check edge indices are within valid range
max_node_id = edge_index.max().item()
min_node_id = edge_index.min().item()

print(f"\nEdge Index Range:")
print(f"  Min node ID: {min_node_id}")
print(f"  Max node ID: {max_node_id}")
print(f"  Num nodes:   {num_nodes}")

assert min_node_id >= 0, "Node IDs should be >= 0"
assert max_node_id < num_nodes, f"Max node ID ({max_node_id}) >= num_nodes ({num_nodes})"

# Check for self-loops
src = edge_index[0]
dst = edge_index[1]
self_loops = (src == dst).sum().item()
print(f"\nSelf-loops: {self_loops:,} ({100 * self_loops / num_edges:.2f}%)")

# Check connected components
unique_src = torch.unique(src).numel()
unique_dst = torch.unique(dst).numel()
connected_nodes = torch.unique(edge_index.flatten()).numel()
isolated_nodes = num_nodes - connected_nodes

print(f"\nConnectivity:")
print(f"  Unique source nodes: {unique_src:,}")
print(f"  Unique dest nodes:   {unique_dst:,}")
print(f"  Connected nodes:     {connected_nodes:,}")
print(f"  Isolated nodes:      {isolated_nodes:,}")

print("\n✓ Graph connectivity is valid")

# ============================================================
# 7. CHECK RAT FEATURES
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
        print(f"  ✓ {feat}")
        print(f"      Range: [{feat_data.min():.4f}, {feat_data.max():.4f}]")
        print(f"      Mean:  {feat_data.mean():.4f}")
    else:
        print(f"  ✗ {feat} - MISSING!")

# Check motif features
expected_motif_features = ["motif_fanin", "motif_fanout", "motif_chain", "motif_cycle"]

print(f"\nMotif Features Check:")
for feat in expected_motif_features:
    if feat in edge_attr_cols:
        idx = edge_attr_cols.index(feat)
        feat_data = edge_attr[:, idx]
        print(f"  ✓ {feat}")
        print(f"      Range: [{feat_data.min():.4f}, {feat_data.max():.4f}]")
        print(f"      Mean:  {feat_data.mean():.4f}")
    else:
        print(f"  ✗ {feat} - MISSING!")

# ============================================================
# 8. LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

# Edge labels
num_laundering_edges = y_edge.sum().item()
pct_laundering_edges = 100 * y_edge.float().mean().item()

print(f"\nEdge Labels:")
print(f"  Laundering: {num_laundering_edges:,} ({pct_laundering_edges:.2f}%)")
print(f"  Normal:     {num_edges - num_laundering_edges:,} ({100 - pct_laundering_edges:.2f}%)")

# Node labels
num_laundering_nodes = y_node.sum().item()
pct_laundering_nodes = 100 * y_node.float().mean().item()

print(f"\nNode Labels:")
print(f"  Laundering: {num_laundering_nodes:,} ({pct_laundering_nodes:.2f}%)")
print(f"  Normal:     {num_nodes - num_laundering_nodes:,} ({100 - pct_laundering_nodes:.2f}%)")

# Check for class imbalance
if pct_laundering_edges < 1:
    print(f"\n⚠ WARNING: Severe class imbalance ({pct_laundering_edges:.2f}% positive)")
    print("  → Consider using weighted loss or oversampling")
elif pct_laundering_edges < 5:
    print(f"\n⚠ Note: Moderate class imbalance ({pct_laundering_edges:.2f}% positive)")

# ============================================================
# 9. TEMPORAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL VALIDATION (for GraphSAGE-T)")
print("=" * 70)

# Check temporal ordering
time_diffs = timestamps[1:] - timestamps[:-1]
negative_gaps = (time_diffs < 0).sum().item()

time_span_seconds = timestamps.max().item() - timestamps.min().item()
time_span_days = time_span_seconds / (3600 * 24)

print(f"\nTimestamp Statistics:")
print(f"  Min timestamp: {timestamps.min().item()}")
print(f"  Max timestamp: {timestamps.max().item()}")
print(f"  Time span:     {time_span_days:.1f} days")
print(f"  Negative gaps: {negative_gaps}")

if negative_gaps > 0:
    print(f"\n❌ ERROR: {negative_gaps} temporal ordering violations!")
    print("  → GraphSAGE-T requires strictly increasing timestamps")
else:
    print("\n✓ Timestamps are in chronological order")

# Check for duplicate timestamps
unique_timestamps = torch.unique(timestamps).numel()
duplicate_ratio = 1 - (unique_timestamps / num_edges)

print(f"\nTimestamp Uniqueness:")
print(f"  Unique timestamps: {unique_timestamps:,}")
print(f"  Duplicate ratio:   {duplicate_ratio*100:.2f}%")

if duplicate_ratio > 0.5:
    print("  ⚠ High duplicate ratio - may need tie-breaking logic")

# ============================================================
# 10. COMPARE WITH GRAPH_STATS.JSON
# ============================================================

print("\n" + "=" * 70)
print("METADATA VALIDATION")
print("=" * 70)

print(f"\nGraph Stats (from graph_stats.json):")
print(f"  Dataset type:           {graph_stats.get('dataset_type', 'N/A')}")
print(f"  Num nodes (recorded):   {graph_stats.get('num_nodes', 'N/A'):,}")
print(f"  Num edges (recorded):   {graph_stats.get('num_edges', 'N/A'):,}")
print(f"  Num edge features:      {graph_stats.get('num_edge_features', 'N/A')}")
print(f"  Num node features:      {graph_stats.get('num_node_features', 'N/A')}")
print(f"  Laundering % (edges):   {graph_stats.get('pct_laundering_edges', 'N/A'):.2f}%")

# Verify consistency
assert graph_stats['num_nodes'] == num_nodes, "Mismatch: num_nodes"
assert graph_stats['num_edges'] == num_edges, "Mismatch: num_edges"
assert graph_stats['num_edge_features'] == edge_attr.shape[1], "Mismatch: num_edge_features"
assert graph_stats['num_node_features'] == x.shape[1], "Mismatch: num_node_features"

print("\n✓ Metadata is consistent with loaded data")

# ============================================================
# 11. VISUALIZATIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("RAT Medium Graph Analysis", fontsize=16, fontweight='bold')

# 1. Degree distribution
degrees = torch.bincount(edge_index[0]).float()
axes[0, 0].hist(degrees.numpy(), bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Degree')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Out-Degree Distribution')
axes[0, 0].set_yscale('log')

# 2. Label distribution
label_counts = [num_edges - num_laundering_edges, num_laundering_edges]
axes[0, 1].bar(['Normal', 'Laundering'], label_counts, color=['green', 'red'], alpha=0.7)
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Edge Label Distribution')
axes[0, 1].set_yscale('log')

# 3. RAT score distribution (if exists)
if 'RAT_score' in edge_attr_cols:
    rat_score_idx = edge_attr_cols.index('RAT_score')
    rat_scores = edge_attr[:, rat_score_idx].numpy()
    axes[0, 2].hist(rat_scores, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 2].set_xlabel('RAT Score')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('RAT Score Distribution')
else:
    axes[0, 2].text(0.5, 0.5, 'RAT_score not found', ha='center', va='center')
    axes[0, 2].set_title('RAT Score Distribution')

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

# 5. Timestamp distribution
time_hours = (timestamps - timestamps.min()).float() / 3600
axes[1, 1].hist(time_hours.numpy(), bins=50, edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Time (hours from start)')
axes[1, 1].set_ylabel('Event Count')
axes[1, 1].set_title('Temporal Distribution')

# 6. RAT component scores (if available)
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
plt.savefig(os.path.join(GRAPH_DIR, 'graph_analysis.png'), dpi=150, bbox_inches='tight')
print("\n✓ Saved visualization: graph_analysis.png")

# ============================================================
# 12. FINAL SUMMARY
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
if negative_gaps > 0:
    issues.append("Temporal ordering violations")
if isolated_nodes > num_nodes * 0.1:
    issues.append(f"High isolated nodes ({isolated_nodes:,})")
if pct_laundering_edges < 1:
    issues.append(f"Severe class imbalance ({pct_laundering_edges:.2f}%)")

if len(issues) == 0:
    print("\n✅ ALL CHECKS PASSED!")
    print("\nThe graph is ready for training:")
    print(f"  • {num_nodes:,} nodes, {num_edges:,} edges")
    print(f"  • {len(edge_attr_cols)} edge features (including RAT & motif)")
    print(f"  • {x.shape[1]} node features")
    print(f"  • {pct_laundering_edges:.2f}% laundering edges")
    print(f"  • Temporal span: {time_span_days:.1f} days")
    print("\n✓ GraphSAGE / GraphSAGE-T ready!")
else:
    print("\n⚠ ISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    print("\nReview issues before training.")

print("\n" + "=" * 70)
print("DEBUGGING COMPLETE")
print("=" * 70)
print(f"\n✓ Debug report saved to: {OUTPUT_FILE}")

# Close logger and restore stdout
sys.stdout = logger.terminal
logger.close()

print(f"\n✓ Full debug report saved to: {OUTPUT_FILE}")

# ============================================================
# CREATE SUMMARY FILE
# ============================================================

SUMMARY_FILE = os.path.join(GRAPH_DIR, "debug_summary.txt")

with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("GRAPH VALIDATION SUMMARY\n")
    f.write("=" * 70 + "\n\n")
    
    f.write(f"Dataset: {os.path.basename(GRAPH_DIR)}\n")
    f.write(f"Type: {graph_stats.get('dataset_type', 'Unknown')}\n\n")
    
    f.write("GRAPH STRUCTURE:\n")
    f.write(f"  Nodes: {num_nodes:,}\n")
    f.write(f"  Edges: {num_edges:,}\n")
    f.write(f"  Avg degree: {num_edges / num_nodes:.2f}\n\n")
    
    f.write("FEATURES:\n")
    f.write(f"  Edge features: {edge_attr.shape[1]} ({len(rat_features)} RAT + {len(motif_features)} motif + {len(structural_features)} structural)\n")
    f.write(f"  Node features: {x.shape[1]}\n\n")
    
    f.write("LABELS:\n")
    f.write(f"  Laundering edges: {num_laundering_edges:,} ({pct_laundering_edges:.2f}%)\n")
    f.write(f"  Laundering nodes: {num_laundering_nodes:,} ({pct_laundering_nodes:.2f}%)\n\n")
    
    f.write("DATA QUALITY:\n")
    f.write(f"  NaN/Inf values: {'✓ None' if len(nan_inf_issues) == 0 else '✗ ' + ', '.join(nan_inf_issues)}\n")
    f.write(f"  Temporal order: {'✓ Valid' if negative_gaps == 0 else '✗ ' + str(negative_gaps) + ' violations'}\n")
    f.write(f"  Isolated nodes: {isolated_nodes:,}\n\n")
    
    f.write("VALIDATION RESULT:\n")
    if len(issues) == 0:
        f.write("  ✅ ALL CHECKS PASSED - Ready for training\n")
    else:
        f.write("  ⚠ ISSUES FOUND:\n")
        for i, issue in enumerate(issues, 1):
            f.write(f"    {i}. {issue}\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write("FILES GENERATED:\n")
    f.write(f"  • {OUTPUT_FILE} (full report)\n")
    f.write(f"  • {SUMMARY_FILE} (this summary)\n")
    f.write(f"  • graph_analysis.png (visualizations)\n")

print(f"✓ Summary saved to: {SUMMARY_FILE}")