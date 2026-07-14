
import os
import argparse
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


BASELINE_COLOR = "#4C72B0"   # plain transaction-derived features (blue)
INJECTED_COLOR = "#8172B3"   # RAT-injector / motif features (purple)


def _feature_category(feature_name):
    """Classify a feature as 'Injected' (RAT/motif signals) or 'Baseline'."""
    if feature_name.startswith("RAT_") or feature_name.startswith("motif_"):
        return "Injected"
    return "Baseline"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        required=True,
        help="Path to the RAT dataset CSV (e.g., RAT-medium).",
    )
    parser.add_argument(
        "--label_col",
        default="Is Laundering",
        help="Name of the label column (default: Is Laundering).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of top features to keep (default: 20).",
    )
    parser.add_argument(
        "--out_dir",
        default="results/feature_importance",
        help="Output directory to save importance files.",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("====================================================")
    print("   FEATURE IMPORTANCE (RandomForest on RAT CSV)     ")
    print("====================================================")
    print(f"[INFO] Loading CSV from: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    print(f"[INFO] Loaded shape: {df.shape}")

    # -----------------------------
    # Preprocess: Create engineered features
    # -----------------------------
    print("\n[INFO] Engineering features from raw columns...")
    
    # 1. Log transforms for amounts
    if 'Amount Received' in df.columns:
        df['log_amt_rec'] = np.log1p(df['Amount Received'])
        print("  ✓ Created log_amt_rec")
    if 'Amount Paid' in df.columns:
        df['log_amt_paid'] = np.log1p(df['Amount Paid'])
        print("  ✓ Created log_amt_paid")
    
    # 2. Binary comparisons
    if 'From Bank' in df.columns and 'To Bank' in df.columns:
        df['same_bank'] = (df['From Bank'] == df['To Bank']).astype(int)
        print("  ✓ Created same_bank")
    if 'Receiving Currency' in df.columns and 'Payment Currency' in df.columns:
        df['same_currency'] = (df['Receiving Currency'] == df['Payment Currency']).astype(int)
        print("  ✓ Created same_currency")
    
    # 3. Temporal features
    if 'hour' in df.columns:
        df['hour_of_day'] = df['hour']
        print("  ✓ Created hour_of_day")
    if 'weekday' in df.columns:
        df['day_of_week'] = df['weekday']
        print("  ✓ Created day_of_week")
    
    # is_weekend
    if 'RAT_is_weekend' in df.columns:
        df['is_weekend'] = df['RAT_is_weekend']
        print("  ✓ Created is_weekend")
    
    # 4. Normalized timestamp
    if 'Timestamp' in df.columns:
        df['Timestamp_dt'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df['ts_normalized'] = (df['Timestamp_dt'] - df['Timestamp_dt'].min()).dt.total_seconds()
        df['ts_normalized'] = df['ts_normalized'] / df['ts_normalized'].max()
        print("  ✓ Created ts_normalized")
    
    # 5. Time since first seen (log transformed) - EXCLUDED per user request
    # if 'src_first_seen' in df.columns and 'Timestamp_dt' in df.columns:
    #     src_first_dt = pd.to_datetime(df['src_first_seen'], errors='coerce')
    #     time_diff = (df['Timestamp_dt'] - src_first_dt).dt.total_seconds()
    #     df['log_time_since_src'] = np.log1p(time_diff.fillna(0).clip(lower=0))
    #     print("  ✓ Created log_time_since_src")
    # 
    # if 'dst_first_seen' in df.columns and 'Timestamp_dt' in df.columns:
    #     dst_first_dt = pd.to_datetime(df['dst_first_seen'], errors='coerce')
    #     time_diff = (df['Timestamp_dt'] - dst_first_dt).dt.total_seconds()
    #     df['log_time_since_dst'] = np.log1p(time_diff.fillna(0).clip(lower=0))
    #     print("  ✓ Created log_time_since_dst")
    
    # 6. One-hot encode Payment Format
    if 'Payment Format' in df.columns:
        pf_dummies = pd.get_dummies(df['Payment Format'], prefix='pf')
        df = pd.concat([df, pf_dummies], axis=1)
        print(f"  ✓ Created {len(pf_dummies.columns)} payment format features")
    
    # 7. One-hot encode Receiving Currency
    if 'Receiving Currency' in df.columns:
        rc_dummies = pd.get_dummies(df['Receiving Currency'], prefix='rc')
        df = pd.concat([df, rc_dummies], axis=1)
        print(f"  ✓ Created {len(rc_dummies.columns)} receiving currency features")
    
    print(f"[INFO] Shape after feature engineering: {df.shape}")

    # -----------------------------
    # Define the 56 features to use (excluded log_time_since_src/dst)
    # -----------------------------
    REQUIRED_FEATURES = [
        "log_amt_rec", "log_amt_paid", "same_bank", "same_currency",
        "hour_of_day", "day_of_week", "is_weekend", "ts_normalized",
        # "log_time_since_src", "log_time_since_dst",  # EXCLUDED
        "pf_ACH", "pf_Bitcoin", "pf_Cash", "pf_Cheque", 
        "pf_Credit Card", "pf_Reinvestment", "pf_Wire",
        "rc_Australian Dollar", "rc_Bitcoin", "rc_Brazil Real", "rc_Canadian Dollar",
        "rc_Euro", "rc_Mexican Peso", "rc_Ruble", "rc_Rupee", "rc_Saudi Riyal",
        "rc_Shekel", "rc_Swiss Franc", "rc_UK Pound", "rc_US Dollar", "rc_Yen", "rc_Yuan",
        "RAT_is_off_hours", "RAT_is_weekend", "RAT_is_cross_bank",
        "RAT_src_amount_z_pos", "RAT_dst_amount_z_pos",
        "RAT_src_out_deg_norm", "RAT_dst_in_deg_norm",
        "RAT_src_burst_norm", "RAT_dst_burst_norm", "RAT_combined_burst",
        "RAT_same_entity", "RAT_src_entity_accounts", "RAT_dst_entity_accounts",
        "RAT_src_entity_acct_norm", "RAT_dst_entity_acct_norm",
        "RAT_src_pattern_flag", "RAT_dst_pattern_flag", "RAT_mutual_flag",
        "motif_fanin", "motif_fanout", "motif_chain", "motif_cycle",
        "RAT_offender_score", "RAT_target_score", 
        "RAT_guardian_weakness_score", "RAT_score"
    ]
    
    # Check which features are available
    feature_cols = [f for f in REQUIRED_FEATURES if f in df.columns]
    missing_features = [f for f in REQUIRED_FEATURES if f not in df.columns]
    
    if missing_features:
        print(f"\n[WARN] {len(missing_features)} required features not found:")
        for feat in missing_features:
            print(f"  - {feat}")
    
    print(f"\n[INFO] Using {len(feature_cols)} / {len(REQUIRED_FEATURES)} features")
    print(f"[INFO] Label column: {args.label_col}")
    
    assert args.label_col in df.columns, f"Label column '{args.label_col}' not found."

    # Extract features and labels
    X = df[feature_cols].fillna(0).values
    y = df[args.label_col].values

    print(f"\n[INFO] Final X shape: {X.shape}")
    print(f"[INFO] Final y shape: {y.shape}")
    print(f"[INFO] Class distribution: {np.bincount(y)}")

    # -----------------------------
    # Train / val / test split
    # -----------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    print("\n[INFO] Splits:")
    print(f"  Train: {X_train.shape[0]} (class 0: {np.sum(y_train==0)}, class 1: {np.sum(y_train==1)})")
    print(f"  Val:   {X_val.shape[0]} (class 0: {np.sum(y_val==0)}, class 1: {np.sum(y_val==1)})")
    print(f"  Test:  {X_test.shape[0]} (class 0: {np.sum(y_test==0)}, class 1: {np.sum(y_test==1)})")

    # -----------------------------
    # Train Random Forest
    # -----------------------------
    print("\n[INFO] Training RandomForest surrogate model...")
    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    print("[INFO] Training complete!")

    # Quick sanity metrics
    val_proba = rf.predict_proba(X_val)[:, 1]
    val_roc = roc_auc_score(y_val, val_proba)
    val_aupr = average_precision_score(y_val, val_proba)
    
    test_proba = rf.predict_proba(X_test)[:, 1]
    test_roc = roc_auc_score(y_test, test_proba)
    test_aupr = average_precision_score(y_test, test_proba)
    
    print(f"\n[INFO] Validation Metrics:")
    print(f"  ROC-AUC: {val_roc:.4f}")
    print(f"  AUPR:    {val_aupr:.4f}")
    
    print(f"\n[INFO] Test Metrics:")
    print(f"  ROC-AUC: {test_roc:.4f}")
    print(f"  AUPR:    {test_aupr:.4f}")

    # -----------------------------
    # Feature importances
    # -----------------------------
    importances = rf.feature_importances_
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print("\n[INFO] Top 20 features (by importance):")
    print(imp_df.head(20).to_string(index=False))

    # Save full ranking
    full_path = os.path.join(args.out_dir, "feature_importances_full.csv")
    imp_df.to_csv(full_path, index=False)
    print(f"\n[INFO] Saved full importances → {full_path}")

    # Save Top-K lists for ablation
    for k in [5, 10, 20]:
        if k <= len(feature_cols):
            topk_df = imp_df.head(k)
            topk_path = os.path.join(args.out_dir, f"top_{k}_features.txt")
            with open(topk_path, "w") as f:
                for feat in topk_df["feature"]:
                    f.write(feat + "\n")
            print(f"[INFO] Saved Top-{k} feature list → {topk_path}")

    # -----------------------------
    # Plot Top-K bar chart
    # -----------------------------
    top_k = min(args.top_k, len(feature_cols))
    topk_df = imp_df.head(top_k)

    plot_df = topk_df.iloc[::-1]  # Reverse for plotting

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    categories = plot_df["feature"].map(_feature_category)
    colors = categories.map({"Baseline": BASELINE_COLOR, "Injected": INJECTED_COLOR})
    bars = ax.barh(
        plot_df["feature"],
        plot_df["importance"],
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )

    max_importance = plot_df["importance"].max()
    for bar, value in zip(bars, plot_df["importance"]):
        ax.text(
            bar.get_width() + max_importance * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#333333",
        )

    ax.set_xlabel("Feature Importance", fontsize=12, labelpad=10)
    ax.set_ylabel("Feature", fontsize=12, labelpad=10)
    ax.set_title(
        f"Top {top_k} RAT Features by RandomForest Importance",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlim(0, max_importance * 1.15)
    ax.grid(axis="x", linestyle="--", alpha=0.5, zorder=0)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=10)

    present_categories = [c for c in ["Baseline", "Injected"] if (categories == c).any()]
    if len(present_categories) > 1:
        legend_colors = {"Baseline": BASELINE_COLOR, "Injected": INJECTED_COLOR}
        handles = [
            Patch(facecolor=legend_colors[c], edgecolor="white", label=c)
            for c in present_categories
        ]
        ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=10)

    fig.tight_layout()

    fig_path = os.path.join(args.out_dir, f"feature_importance_top_{top_k}.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"\n[INFO] Saved top-{top_k} bar chart → {fig_path}")

    print("\n[SUCCESS] Feature importance analysis complete.")
    print(f"\n📊 Summary:")
    print(f"  - Features analyzed: {len(feature_cols)}")
    print(f"  - Top feature: {imp_df.iloc[0]['feature']} ({imp_df.iloc[0]['importance']:.4f})")
    print(f"  - Validation ROC-AUC: {val_roc:.4f} | AUPR: {val_aupr:.4f}")
    print(f"  - Test ROC-AUC: {test_roc:.4f} | AUPR: {test_aupr:.4f}")


if __name__ == "__main__":
    main()