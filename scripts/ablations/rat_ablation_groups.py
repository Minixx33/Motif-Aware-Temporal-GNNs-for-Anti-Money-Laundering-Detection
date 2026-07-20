# ablation_groups.py

# NOTE: motif_graph_builder_static.py encodes payment_format / receiving_currency
# as single integer category-code columns ("pf_code", "rc_code") for the GNN's
# embedding lookup, NOT as one-hot dummies (see lines ~309-310 of that file).
# This list must match the graph's actual edge_attr_cols.json schema, since
# run_ablation_static.py subsets edge_attr.pt by these exact names.
FULL_FEATURES = [
    "log_amt_rec", "log_amt_paid",
    "same_bank", "same_currency",
    "hour_of_day", "day_of_week", "is_weekend",
    "ts_normalized", "log_time_since_src", "log_time_since_dst",
    "pf_code", "rc_code",
    "RAT_is_off_hours","RAT_is_weekend","RAT_is_cross_bank",
    "RAT_src_amount_z_pos","RAT_dst_amount_z_pos",
    "RAT_src_out_deg_norm","RAT_dst_in_deg_norm",
    "RAT_src_burst_norm","RAT_dst_burst_norm","RAT_combined_burst",
    "RAT_same_entity",
    "RAT_src_entity_accounts","RAT_dst_entity_accounts",
    "RAT_src_entity_acct_norm","RAT_dst_entity_acct_norm",
    "RAT_src_pattern_flag","RAT_dst_pattern_flag","RAT_mutual_flag",
    "motif_fanin","motif_fanout","motif_chain","motif_cycle",
    "RAT_offender_score","RAT_target_score",
    "RAT_guardian_weakness_score","RAT_score"
]

# --- 8 Ablations ---

NO_STRUCT = [
    "RAT_src_out_deg_norm", "RAT_dst_in_deg_norm"
]

NO_TEMP = [
    "hour_of_day","day_of_week","is_weekend",
    "ts_normalized","log_time_since_src","log_time_since_dst",
    "RAT_is_off_hours","RAT_is_weekend"
]

NO_AMOUNT = [
    "log_amt_rec","log_amt_paid",
    "RAT_src_amount_z_pos","RAT_dst_amount_z_pos"
]

NO_BURST_PATTERN = [
    "RAT_src_burst_norm","RAT_dst_burst_norm","RAT_combined_burst",
    "RAT_src_pattern_flag","RAT_dst_pattern_flag","RAT_mutual_flag"
]

NO_ENTITY = [
    "RAT_same_entity",
    "RAT_src_entity_accounts","RAT_dst_entity_accounts",
    "RAT_src_entity_acct_norm","RAT_dst_entity_acct_norm"
]

NO_RAT_SCORES = [
    "RAT_offender_score","RAT_target_score",
    "RAT_guardian_weakness_score","RAT_score"
]

NO_MOTIF = [
    "motif_fanin","motif_fanout","motif_chain","motif_cycle"
]

NO_CROSS_BANK = [
    "RAT_is_cross_bank"
]

TOP20_FEATURES = [
    "pf_ACH",
    "pf_Cheque",
    "log_amt_paid",
    "log_amt_rec",
    "RAT_dst_in_deg_norm",
    "pf_Credit Card",
    "motif_fanin",
    "motif_cycle",
    "RAT_combined_burst",
    "ts_normalized",
    "RAT_src_burst_norm",
    "RAT_src_out_deg_norm",
    "motif_fanout",
    "RAT_target_score",
    "RAT_score",
    "motif_chain",
    "RAT_dst_burst_norm",
    "RAT_dst_entity_acct_norm",
    "RAT_offender_score",
    "RAT_dst_entity_accounts",
]

ABLATED_SETS = {
    "no_struct": NO_STRUCT,
    "no_temp": NO_TEMP,
    "no_amount": NO_AMOUNT,
    "no_burst_pattern": NO_BURST_PATTERN,
    "no_entity": NO_ENTITY,
    "no_rat_scores": NO_RAT_SCORES,
    "no_motif": NO_MOTIF,
    "no_crossbank": NO_CROSS_BANK,
    "top20_features": TOP20_FEATURES
}
