import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
from matplotlib.patches import Patch


BASELINE_COLOR = "#B39DDB"   # plain transaction-derived features (light purple)
INJECTED_COLOR = "#4A148C"   # motif / SLT-injector features (dark purple)
CUTOFF_LABEL_COLOR = "#1A052E"  # near-black purple used for the cut-off arrow + value
Y_AXIS_CAP = 0.085  # hard y-axis ceiling; any bar taller than this gets cut off

# Explicitly register Times New Roman with matplotlib's font manager. Relying
# on rcParams alone can silently fall back to a different serif font if the
# font manager's cache was built before Times New Roman was discoverable.
for _font_path in (
    r"C:\Windows\Fonts\times.ttf",
    r"C:\Windows\Fonts\Times New Roman.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
):
    if os.path.exists(_font_path):
        fm.fontManager.addfont(_font_path)
        break

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split


BASE_FEATURES = [
    "log_amt_rec",
    "log_amt_paid",
    "same_bank",
    "same_currency",
    "hour_of_day",
    "day_of_week",
    "is_off_hours",
    "is_weekend",
    "ts_normalized",
]

MOTIF_FEATURES = [
    "motif_fanin",
    "motif_fanout",
    "motif_chain",
    "motif_cycle",
]

# These describe which rows/peers were selected by the injector rather than
# transaction-level SLT exposure. They must not be used as model inputs.
SLT_EXCLUDED_COLUMNS = {
    "SLT_injected",
    "SLT_intensity_level",
    "src_is_high_risk_peer",
    "dst_is_high_risk_peer",
}

RAW_EXCLUDED_COLUMNS = {
    "Timestamp",
    "Timestamp_dt",
    "date_only",
    "From Bank",
    "To Bank",
    "Account",
    "Account.1",
    "Receiving Currency",
    "Payment Currency",
    "Payment Format",
    "Amount Received",
    "Amount Paid",
    "Is Laundering",
    "Is_Laundering",
}


def resolve_csv_path(path_value: str, slt_level: str) -> Path:
    """Accept either a CSV file or a directory containing an SLT dataset CSV."""
    path = Path(path_value).expanduser()

    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Input file is not a CSV: {path}")
        return path

    if not path.is_dir():
        raise FileNotFoundError(f"CSV path does not exist: {path}")

    level = slt_level.lower()
    candidates = sorted(
        p for p in path.rglob("*.csv")
        if "slt" in p.name.lower() and level in p.name.lower()
    )

    if not candidates:
        all_slt = sorted(p for p in path.rglob("*.csv") if "slt" in p.name.lower())
        if len(all_slt) == 1:
            return all_slt[0]
        if all_slt:
            names = "\n  - ".join(str(p) for p in all_slt[:20])
            raise RuntimeError(
                f"No unique SLT '{slt_level}' CSV was found in {path}. "
                f"Available SLT CSVs include:\n  - {names}"
            )
        raise RuntimeError(
            f"No SLT dataset CSV was found under {path}. "
            "The feature-importance input must be the injected transaction CSV, "
            "not a model-metrics/results CSV."
        )

    if len(candidates) > 1:
        names = "\n  - ".join(str(p) for p in candidates[:20])
        raise RuntimeError(
            f"Multiple SLT '{slt_level}' CSVs were found. Pass the exact file path:\n  - {names}"
        )

    return candidates[0]


def resolve_label_column(df: pd.DataFrame, requested: str) -> str:
    candidates = [requested, "Is Laundering", "Is_Laundering", "is_laundering"]
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        f"Label column not found. Tried: {candidates}. "
        f"Available columns include: {list(df.columns[:30])}"
    )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[INFO] Engineering baseline transaction features...")

    if "Amount Received" in df.columns:
        values = pd.to_numeric(df["Amount Received"], errors="coerce").fillna(0).clip(lower=0)
        df["log_amt_rec"] = np.log1p(values)
        print("  [OK] Created log_amt_rec")

    if "Amount Paid" in df.columns:
        values = pd.to_numeric(df["Amount Paid"], errors="coerce").fillna(0).clip(lower=0)
        df["log_amt_paid"] = np.log1p(values)
        print("  [OK] Created log_amt_paid")

    if {"From Bank", "To Bank"}.issubset(df.columns):
        df["same_bank"] = (df["From Bank"] == df["To Bank"]).astype(np.int8)
        print("  [OK] Created same_bank")
    elif "is_cross_bank" in df.columns:
        df["same_bank"] = 1 - pd.to_numeric(df["is_cross_bank"], errors="coerce").fillna(0)
        print("  [OK] Created same_bank from is_cross_bank")

    if {"Receiving Currency", "Payment Currency"}.issubset(df.columns):
        df["same_currency"] = (
            df["Receiving Currency"] == df["Payment Currency"]
        ).astype(np.int8)
        print("  [OK] Created same_currency")

    if "hour" in df.columns:
        df["hour_of_day"] = pd.to_numeric(df["hour"], errors="coerce")
        print("  [OK] Created hour_of_day")

    if "weekday" in df.columns:
        df["day_of_week"] = pd.to_numeric(df["weekday"], errors="coerce")
        print("  [OK] Created day_of_week")

    if "is_off_hours" in df.columns:
        df["is_off_hours"] = pd.to_numeric(df["is_off_hours"], errors="coerce")
    elif "hour_of_day" in df.columns:
        df["is_off_hours"] = ((df["hour_of_day"] < 6) | (df["hour_of_day"] >= 22)).astype(np.int8)
        print("  [OK] Created is_off_hours")

    if "is_weekend" in df.columns:
        df["is_weekend"] = pd.to_numeric(df["is_weekend"], errors="coerce")
    elif "day_of_week" in df.columns:
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.int8)
        print("  [OK] Created is_weekend")

    if "Timestamp" in df.columns:
        timestamp = pd.to_datetime(df["Timestamp"], errors="coerce")
        seconds = (timestamp - timestamp.min()).dt.total_seconds()
        denominator = seconds.max()
        if pd.notna(denominator) and denominator > 0:
            df["ts_normalized"] = seconds / denominator
        else:
            df["ts_normalized"] = 0.0
        print("  [OK] Created ts_normalized")

    if "Payment Format" in df.columns:
        pf_dummies = pd.get_dummies(df["Payment Format"], prefix="pf", dtype=np.int8)
        df = pd.concat([df, pf_dummies], axis=1)
        print(f"  [OK] Created {len(pf_dummies.columns)} payment-format features")

    if "Receiving Currency" in df.columns:
        rc_dummies = pd.get_dummies(df["Receiving Currency"], prefix="rc", dtype=np.int8)
        df = pd.concat([df, rc_dummies], axis=1)
        print(f"  [OK] Created {len(rc_dummies.columns)} receiving-currency features")

    return df


def is_slt_feature(column: str) -> bool:
    if column in SLT_EXCLUDED_COLUMNS:
        return False

    lower = column.lower()

    # Explicit SLT-named columns, including src_SLT_* and dst_SLT_*.
    if "slt" in lower:
        return True

    # Injector-generated peer/tie exposure columns whose names do not contain SLT.
    peer_or_tie_tokens = (
        "peer_risk",
        "high_risk_neighbor",
        "total_neighbor",
        "high_risk_tie",
        "total_tie",
        "strong_ties",
    )
    return lower.startswith(("src_", "dst_")) and any(
        token in lower for token in peer_or_tie_tokens
    )


def _feature_category(feature_name: str) -> str:
    """Classify a feature as 'Injected' (motif/SLT signals) or 'Baseline'."""
    if feature_name in MOTIF_FEATURES or is_slt_feature(feature_name):
        return "Injected"
    return "Baseline"


def select_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    payment_features = sorted(c for c in df.columns if c.startswith("pf_"))
    currency_features = sorted(c for c in df.columns if c.startswith("rc_"))
    slt_features = sorted(c for c in df.columns if is_slt_feature(c))

    requested = BASE_FEATURES + payment_features + currency_features + MOTIF_FEATURES + slt_features

    # Preserve order while removing duplicates.
    feature_cols = []
    seen = set()
    for col in requested:
        if col in seen or col not in df.columns:
            continue
        if col in RAW_EXCLUDED_COLUMNS or col in SLT_EXCLUDED_COLUMNS:
            continue
        feature_cols.append(col)
        seen.add(col)

    if not slt_features:
        raise RuntimeError(
            "No SLT feature columns were detected. Confirm that --csv_path points to "
            "an injected SLT transaction dataset rather than a training-results CSV."
        )

    return feature_cols, slt_features



def columns_needed_for_analysis(all_columns: list[str], label_col: str) -> list[str]:
    """Read only columns needed to engineer and select the model inputs."""
    raw_baseline = [
        label_col,
        "Amount Received",
        "Amount Paid",
        "From Bank",
        "To Bank",
        "Receiving Currency",
        "Payment Currency",
        "Payment Format",
        "hour",
        "weekday",
        "is_off_hours",
        "is_weekend",
        "Timestamp",
        "is_cross_bank",
    ]
    theory_columns = [
        col for col in all_columns
        if col in MOTIF_FEATURES or is_slt_feature(col)
    ]

    requested = raw_baseline + theory_columns
    available = []
    seen = set()
    for col in requested:
        if col in all_columns and col not in seen:
            available.append(col)
            seen.add(col)
    return available


def _allocated_take(
    class_rows_in_chunk: int,
    class_rows_remaining: int,
    target_remaining: int,
) -> int:
    """Allocate an exact final sample across chunks without retaining all rows."""
    if class_rows_in_chunk <= 0 or target_remaining <= 0:
        return 0
    if class_rows_remaining <= class_rows_in_chunk:
        return min(class_rows_in_chunk, target_remaining)
    proportional = int(round(target_remaining * class_rows_in_chunk / class_rows_remaining))
    return min(class_rows_in_chunk, target_remaining, max(0, proportional))


def load_analysis_rows(
    csv_path: Path,
    label_col: str,
    read_columns: list[str],
    max_rows: int,
    chunksize: int,
    random_state: int,
) -> tuple[pd.DataFrame, dict]:
    """
    Load the selected columns.

    When max_rows > 0, use two chunked passes to retain an exact-size stratified
    sample. All positives are retained whenever they fit; the remaining capacity
    is filled with randomly sampled negatives.
    """
    if max_rows <= 0:
        print("[INFO] Reading the full dataset using only required columns...")
        df = pd.read_csv(
            csv_path,
            usecols=read_columns,
            low_memory=False,
            memory_map=True,
        )
        return df, {
            "sampling_applied": False,
            "source_rows": int(len(df)),
            "sampled_rows": int(len(df)),
        }

    print(
        f"[INFO] Counting labels in chunks of {chunksize:,} rows before sampling..."
    )
    class_counts = {0: 0, 1: 0}
    invalid_labels = 0
    for label_chunk in pd.read_csv(
        csv_path,
        usecols=[label_col],
        chunksize=chunksize,
        low_memory=True,
    ):
        labels = pd.to_numeric(label_chunk[label_col], errors="coerce")
        invalid_labels += int(labels.isna().sum())
        valid = labels.dropna().astype(np.int8)
        counts = valid.value_counts()
        class_counts[0] += int(counts.get(0, 0))
        class_counts[1] += int(counts.get(1, 0))

    source_rows = class_counts[0] + class_counts[1]
    if source_rows == 0:
        raise ValueError("No valid binary labels were found in the CSV.")

    target_total = min(max_rows, source_rows)
    target_pos = min(class_counts[1], target_total)
    target_neg = min(class_counts[0], target_total - target_pos)

    # This only matters in the unlikely case that positives exceed max_rows.
    if target_pos + target_neg < target_total:
        target_pos = min(class_counts[1], target_total - target_neg)

    print(
        "[INFO] Source class counts: "
        f"0={class_counts[0]:,}, 1={class_counts[1]:,}, invalid={invalid_labels:,}"
    )
    print(
        "[INFO] Sampling target: "
        f"0={target_neg:,}, 1={target_pos:,}, total={target_neg + target_pos:,}"
    )

    rng = np.random.default_rng(random_state)
    remaining_total = {0: class_counts[0], 1: class_counts[1]}
    remaining_target = {0: target_neg, 1: target_pos}
    sampled_parts = []

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            csv_path,
            usecols=read_columns,
            chunksize=chunksize,
            low_memory=True,
        ),
        start=1,
    ):
        labels = pd.to_numeric(chunk[label_col], errors="coerce")
        selected_parts = []

        for class_value in (0, 1):
            class_part = chunk.loc[labels == class_value]
            n_available = len(class_part)
            n_take = _allocated_take(
                n_available,
                remaining_total[class_value],
                remaining_target[class_value],
            )

            if n_take == n_available:
                selected = class_part
            elif n_take > 0:
                selected = class_part.sample(
                    n=n_take,
                    random_state=int(rng.integers(0, 2**31 - 1)),
                )
            else:
                selected = None

            if selected is not None and len(selected):
                selected_parts.append(selected)

            remaining_total[class_value] -= n_available
            remaining_target[class_value] -= n_take

        if selected_parts:
            sampled_parts.append(pd.concat(selected_parts, ignore_index=True))

        if chunk_number % 10 == 0:
            kept = sum(len(part) for part in sampled_parts)
            print(f"  [INFO] Processed {chunk_number} chunks; retained {kept:,} rows")

    if remaining_target[0] != 0 or remaining_target[1] != 0:
        raise RuntimeError(
            "Chunked sampling did not reach the requested class targets: "
            f"remaining={remaining_target}"
        )

    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return sampled, {
        "sampling_applied": True,
        "source_rows": int(source_rows),
        "source_class_0": int(class_counts[0]),
        "source_class_1": int(class_counts[1]),
        "sampled_rows": int(len(sampled)),
        "sampled_class_0": int(target_neg),
        "sampled_class_1": int(target_pos),
        "max_rows": int(max_rows),
        "chunksize": int(chunksize),
        "sampling_random_state": int(random_state),
    }

def _draw_bars_with_smart_labels(ax, bars, values, colors, value_fontsize=11, axis_cap=Y_AXIS_CAP):
    """
    Label each vertical bar with its importance value. The y-axis is held
    to a fixed ceiling (axis_cap); any bar taller than that is cut off right
    at the ceiling, and instead of stretching the chart to fit it, a short
    arrow is drawn straight up ON the bar itself (near its cut-off top),
    with the bar's true value labeled directly on the bar in a dark,
    white-outlined color so it stays legible against either bar shade.

    Returns axis_cap, to pass straight to ax.set_ylim(0, limit).
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return axis_cap

    outline = [pe.withStroke(linewidth=2.5, foreground="white")]

    for bar, value in zip(bars, values):
        x = bar.get_x() + bar.get_width() / 2
        if value > axis_cap:
            bar.set_height(axis_cap)

            # Remove only the top outline while preserving left/right/bottom edges
            bar.set_edgecolor("black")
            bar.set_linewidth(0.9)

            ax.plot(
                [bar.get_x(), bar.get_x() + bar.get_width()],
                [axis_cap, axis_cap],
                color="white",
                linewidth=2.5,
                zorder=6,
            )

            ann = ax.annotate(
                "",
                xy=(x, axis_cap * 0.99),
                xytext=(x, axis_cap * 0.82),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=CUTOFF_LABEL_COLOR,
                    lw=2.5,
                    mutation_scale=18,
                ),
                zorder=5,
            )
            ann.arrow_patch.set_path_effects(outline)

            # Value directly under the arrow
            ax.text(
                x,
                axis_cap * 0.72,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=value_fontsize + 1,
                fontweight="bold",
                color=CUTOFF_LABEL_COLOR,
                rotation=90,
                zorder=6,
                path_effects=outline,
            )
        else:
            ax.text(
                x, value + axis_cap * 0.02, f"{value:.3f}",
                ha="center", va="bottom",
                fontsize=value_fontsize, color="#222222",
                rotation=90,
            )

    return axis_cap


def _plot_feature_importance(plot_df: pd.DataFrame, title: str, ylabel: str, out_path: Path) -> None:
    """Render a styled vertical bar chart of feature importances (plotting only)."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    fig, ax = plt.subplots(figsize=(10, 10))

    categories = plot_df["feature"].map(_feature_category)
    colors = categories.map({"Baseline": BASELINE_COLOR, "Injected": INJECTED_COLOR})
    bars = ax.bar(
        plot_df["feature"],
        plot_df["importance"],
        color=colors,
        edgecolor="black",
        linewidth=0.9,
        zorder=3,
    )

    ylim_upper = _draw_bars_with_smart_labels(
        ax, bars, plot_df["importance"].to_numpy(), colors.to_numpy()
    )

    ax.set_xlabel("Feature", fontsize=15, labelpad=14)
    ax.set_ylabel(plt.ylabel, fontsize=15, labelpad=18)
    ax.set_title(title, fontsize=17, fontweight="bold", pad=22)
    ax.set_ylim(0, ylim_upper)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=13)
    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    present_categories = [c for c in ["Baseline", "Injected"] if (categories == c).any()]
    if len(present_categories) > 1:
        legend_colors = {"Baseline": BASELINE_COLOR, "Injected": INJECTED_COLOR}
        handles = [
            Patch(facecolor=legend_colors[c], edgecolor="black", label=c)
            for c in present_categories
        ]
        ax.legend(
            handles=handles,
            loc="upper right",
            bbox_to_anchor=(0.93, 0.90),
            frameon=True,
            framealpha=0.9,
            fontsize=14,
        )

    fig.tight_layout(pad=2.0)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Random-Forest surrogate feature importance for the SLT AML dataset."
    )
    parser.add_argument(
        "--csv_path",
        required=True,
        help=(
            "Exact SLT transaction CSV, or a directory containing it. "
            "A model-results CSV is not sufficient."
        ),
    )
    parser.add_argument(
        "--slt_level",
        choices=["low", "medium", "high"],
        default="medium",
        help="SLT injection level used when --csv_path is a directory (default: medium).",
    )
    parser.add_argument(
        "--label_col",
        default="Is Laundering",
        help="Label-column name (default: Is Laundering).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of features displayed in the chart (default: 20).",
    )
    parser.add_argument(
        "--out_dir",
        default="results/feature_importance_slt",
        help="Directory for CSV, TXT, JSON, and PNG outputs.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help=(
            "Maximum rows retained through chunked stratified sampling. "
            "All positives are kept when possible. Use 0 for the complete CSV "
            "(default: 0)."
        ),
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="CSV rows processed per chunk when --max_rows is used (default: 100000).",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Parallel Random Forest workers (default: -1, all cores).",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=300,
        help="Number of Random Forest trees (default: 300).",
    )
    args = parser.parse_args()

    csv_path = resolve_csv_path(args.csv_path, args.slt_level)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("====================================================")
    print("   FEATURE IMPORTANCE (RandomForest on SLT CSV)     ")
    print("====================================================")
    print(f"[INFO] Loading CSV from: {csv_path}")
    header = pd.read_csv(csv_path, nrows=0)
    label_col = resolve_label_column(header, args.label_col)
    read_columns = columns_needed_for_analysis(list(header.columns), label_col)
    print(
        f"[INFO] Reading {len(read_columns)} required columns out of "
        f"{len(header.columns)} total columns"
    )
    df, sampling_info = load_analysis_rows(
        csv_path=csv_path,
        label_col=label_col,
        read_columns=read_columns,
        max_rows=args.max_rows,
        chunksize=args.chunksize,
        random_state=42,
    )
    print(f"[INFO] Loaded analysis shape: {df.shape}")

    df = engineer_features(df)
    print(f"[INFO] Shape after feature engineering: {df.shape}")

    feature_cols, detected_slt_features = select_features(df)

    print(f"\n[INFO] Label column: {label_col}")
    print(f"[INFO] Total selected features: {len(feature_cols)}")
    print(f"[INFO] Detected SLT features: {len(detected_slt_features)}")
    for feature in detected_slt_features:
        excluded_note = " [EXCLUDED CONTROL]" if feature in SLT_EXCLUDED_COLUMNS else ""
        print(f"  - {feature}{excluded_note}")

    selected_path = out_dir / "selected_features.txt"
    selected_path.write_text("\n".join(feature_cols) + "\n", encoding="utf-8")

    slt_path = out_dir / "detected_slt_features.txt"
    slt_path.write_text("\n".join(detected_slt_features) + "\n", encoding="utf-8")

    X_df = df[feature_cols].copy()
    for col in feature_cols:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    y = pd.to_numeric(df[label_col], errors="coerce")
    valid_mask = y.notna()
    X = X_df.loc[valid_mask].to_numpy(dtype=np.float32, copy=False)
    y = y.loc[valid_mask].astype(np.int8).to_numpy()

    unique_classes = np.unique(y)
    if not np.array_equal(unique_classes, np.array([0, 1])):
        raise ValueError(f"Expected binary labels 0/1, found: {unique_classes.tolist()}")

    class_counts = np.bincount(y, minlength=2)
    print(f"\n[INFO] Final X shape: {X.shape}")
    print(f"[INFO] Final y shape: {y.shape}")
    print(f"[INFO] Class distribution: 0={class_counts[0]}, 1={class_counts[1]}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42,
    )

    print("\n[INFO] Splits:")
    print(f"  Train: {len(y_train):,} (0={np.sum(y_train == 0):,}, 1={np.sum(y_train == 1):,})")
    print(f"  Val:   {len(y_val):,} (0={np.sum(y_val == 0):,}, 1={np.sum(y_val == 1):,})")
    print(f"  Test:  {len(y_test):,} (0={np.sum(y_test == 0):,}, 1={np.sum(y_test == 1):,})")

    print("\n[INFO] Training RandomForest surrogate model...")
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        class_weight="balanced_subsample",
        max_depth=None,
        n_jobs=args.n_jobs,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    print("[INFO] Training complete.")

    val_proba = rf.predict_proba(X_val)[:, 1]
    test_proba = rf.predict_proba(X_test)[:, 1]

    metrics = {
        "csv_path": str(csv_path),
        "label_col": label_col,
        "n_rows": int(len(y)),
        "n_features": int(len(feature_cols)),
        "n_slt_features_detected": int(len(detected_slt_features)),
        "n_estimators": int(args.n_estimators),
        "n_jobs": int(args.n_jobs),
        "sampling": sampling_info,
        "validation_roc_auc": float(roc_auc_score(y_val, val_proba)),
        "validation_aupr": float(average_precision_score(y_val, val_proba)),
        "test_roc_auc": float(roc_auc_score(y_test, test_proba)),
        "test_aupr": float(average_precision_score(y_test, test_proba)),
    }

    print("\n[INFO] Validation metrics:")
    print(f"  ROC-AUC: {metrics['validation_roc_auc']:.4f}")
    print(f"  AUPR:    {metrics['validation_aupr']:.4f}")
    print("\n[INFO] Test metrics:")
    print(f"  ROC-AUC: {metrics['test_roc_auc']:.4f}")
    print(f"  AUPR:    {metrics['test_aupr']:.4f}")

    imp_df = pd.DataFrame(
        {"feature": feature_cols, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False, ignore_index=True)

    print("\n[INFO] Top 20 features:")
    print(imp_df.head(20).to_string(index=False))

    imp_df.to_csv(out_dir / "feature_importances_full.csv", index=False)
    core_slt_mask = (
        imp_df["feature"].str.contains("SLT", case=False, regex=False)
        & ~imp_df["feature"].isin(["SLT_injected", "SLT_intensity_level"])
    )

    core_slt_df = (
        imp_df.loc[core_slt_mask]
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    core_slt_df.to_csv(
        out_dir / "feature_importances_core_slt.csv",
        index=False
    )

    core_top_k = min(args.top_k, len(core_slt_df))
    core_plot_df = core_slt_df.head(core_top_k)  # already sorted descending by importance

    _plot_feature_importance(
        core_plot_df,
        title=f"Top {core_top_k} Core SLT Features by RandomForest Importance",
        ylabel="SLT Feature",
        out_path=out_dir / f"feature_importance_core_slt_top_{core_top_k}.png",
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    for k in (5, 10, 20):
        if k <= len(imp_df):
            (out_dir / f"top_{k}_features.txt").write_text(
                "\n".join(imp_df.head(k)["feature"]) + "\n",
                encoding="utf-8",
            )

    top_k = min(args.top_k, len(imp_df))
    plot_df = imp_df.head(top_k)  # already sorted descending by importance

    _plot_feature_importance(
        plot_df,
        title=f"Top {top_k} SLT Features by RandomForest Importance",
        ylabel="Feature",
        out_path=out_dir / f"feature_importance_top_{top_k}.png",
    )

    print("\n[SUCCESS] SLT feature-importance analysis complete.")
    print(f"[INFO] Outputs saved to: {out_dir.resolve()}")
    print(
        f"[INFO] Top feature: {imp_df.iloc[0]['feature']} "
        f"({imp_df.iloc[0]['importance']:.6f})"
    )


if __name__ == "__main__":
    main()
