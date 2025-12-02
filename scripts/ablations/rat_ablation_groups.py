# ablation_groups.py

FULL_FEATURES = [
    "log_amt_rec", "log_amt_paid",
    "same_bank", "same_currency",
    "hour_of_day", "day_of_week", "is_weekend",
    "ts_normalized", "log_time_since_src", "log_time_since_dst",
    "pf_ACH","pf_Bitcoin","pf_Cash","pf_Cheque","pf_Credit Card",
    "pf_Reinvestment","pf_Wire",
    "rc_Australian Dollar","rc_Bitcoin","rc_Brazil Real","rc_Canadian Dollar",
    "rc_Euro","rc_Mexican Peso","rc_Ruble","rc_Rupee","rc_Saudi Riyal",
    "rc_Shekel","rc_Swiss Franc","rc_UK Pound","rc_US Dollar","rc_Yen","rc_Yuan",
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
    "pf_Credit Card",
    "motif_cycle",
    "RAT_combined_burst",
    "ts_normalized",
    "RAT_src_burst_norm",
    "RAT_dst_in_deg_norm",
    "RAT_target_score",
    "motif_fanout",
    "motif_fanin",
    "RAT_dst_burst_norm",
    "RAT_src_out_deg_norm",
    "RAT_score",
    "motif_chain",
    "RAT_offender_score",
    "RAT_dst_entity_acct_norm",
    "RAT_guardian_weakness_score",
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
