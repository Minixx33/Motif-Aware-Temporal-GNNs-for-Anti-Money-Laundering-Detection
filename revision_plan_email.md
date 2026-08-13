Subject: AML Paper — Revision Plan in Response to Your Feedback

Dear Professor [Name],

Thank you for the detailed review — it caught a real methodological problem at the core of our injection framework, and we wanted to lay out our plan before we start executing so you can weigh in on priority given our timeline.

**Already fixed.** We confirmed the test-set contamination you flagged: the injection was happening before the train/val/test split, so some test-set laundering transactions had artificially boosted features. We measured the effect directly — reverting only the test-set rows to their pristine values dropped AUPR by about 16% for RAT and 22% for SLT on average across seeds, while ROC-AUC barely moved. We've since rebuilt every RAT/SLT graph so boosting only ever touches training-split rows; validation and test are permanently held at pristine values.

**Primary experimental redesign.** We're adding the natural-features experiment you described: base transaction features vs. base+structural vs. base+RAT vs. base+SLT, using RAT/SLT features computed the normal way from historical data, with no label-conditioned boosting at all. This becomes our primary evidence. The existing low/medium/high injection sweep is being reframed as an explicitly secondary, controlled sensitivity study rather than the paper's central claim.

**Additional leakage we found while reviewing this.** RAT's pattern-flag indicators, and a component of SLT's peer-risk score, both read the AMLworld simulator's ground-truth laundering-pattern file — information a real investigator wouldn't have ahead of a prediction. We're removing both and rebuilding.

**Splitting.** We're adding a chronological split for GraphSAGE and GraphSAGE-T so all three models are evaluated on the same time-ordered protocol as DyRep-Lite, instead of the current stratified-random split.

**Falsification controls.** We're implementing three of the seven controls you listed: a random-selection injection variant (in place of score-ranked selection), a permutation of RAT/SLT features across transactions, and randomized theory weights compared against our current hand-set weights. The remaining two — random features with matched marginal distributions, and matched comparisons controlling for amount/time/degree — need more careful statistical design, so we'd like to scope those as a second pass rather than rush them.

**Baselines.** We're adding logistic regression, Random Forest, XGBoost/LightGBM, and an MLP trained on the existing tabular edge-feature matrix, to test whether the GNNs add value beyond engineered features alone.

**Reporting additions.** Bootstrap confidence intervals across our five seeds, precision-recall curves, calibration diagrams, and recall at fixed precision/false-positive budgets — all computable from data we've already saved, no retraining required.

**Interpretability framing.** We're renaming the Random Forest analysis "standalone feature relevance" rather than describing it as GNN interpretability, and adding permutation importance computed directly on the trained GNNs.

**What we'd like to scope as future work, pending your input:**
1. Recomputing every account-level statistic (degree, entity counts, peer-risk) on a strict point-in-time basis so no feature can reflect future or test-period activity. This is the largest remaining methodological gap and touches nearly every engineered feature in the pipeline — a full fix is a substantial rewrite.
2. True external validation on an independent dataset. As a nearer-term partial step, we can rerun the pipeline on a different IBM AMLworld configuration (e.g., a different size/ratio setting) to at least show the features aren't artifacts of one specific configuration.
3. New architectures (a directed-multigraph-aware GNN, a full-memory temporal GNN beyond DyRep-Lite).

We wanted to flag scope now rather than after the fact — in particular, whether item 1 above needs to be resolved before submission or can be presented as a disclosed limitation with a concrete plan. Happy to discuss whichever way you think is best.

Best regards,
Yasmine Tohamy
(on behalf of Kenzy Khalifa and Afia Murtaza)
