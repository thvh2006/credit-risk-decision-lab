# Feature provenance and decision-time policy

This register explains where portfolio-facing features come from and what was known at the decision point. The machine-readable source is [`reports/feature_provenance.csv`](../reports/feature_provenance.csv).

| Feature | Family | Raw source | Transformation | Cutoff | Fit scope | Role | Main risk |
|---|---|---|---|---|---|---|---|
| numeric_static | application static | train_static_0_*.parquet | Numeric fields eligible under frozen manifest | at date_decision | development | model | medium: definitions reviewed for post-decision semantics |
| categorical_static | application static | train_static_0_* and train_static_cb_0 | Low-cardinality categories eligible under frozen manifest | at date_decision | development | model | medium |
| relative_date_D | date recency | source columns ending D | date_decision - source date in days | at date_decision | development | model | low after relative conversion |
| missing_indicator | coverage | selected source fields | 1 when source value is missing | at date_decision | development | linear baseline | medium: provider/product eligibility signal |
| quantile_bin | nonlinear baseline | eligible numeric features | Ten development-fitted quantile bins with one-hot encoding | at date_decision | development | binned-logit baseline | low |
| case_id | identifier | train_base.case_id | No transformation | at decision | none | excluded | high: identifier leakage |
| WEEK_NUM/MONTH/date_decision | partition/time | train_base | Used only for chronological partitions | at decision | none | excluded | high: direct cohort leakage |
| sensitive-token fields | protected/proxy | source fields matching birth/marital/education/gender/sex | No transformation | at decision | development | excluded | high governance risk |

## Governance rules

1. Outcomes and post-decision events are targets or evaluation evidence, never predictors.
2. Learned bins, imputers, encoders, cutoffs, and feature eligibility are fitted before locked evaluation.
3. Anonymous fields are described by source and statistical behavior; business meanings are not invented.
4. Proxy and fairness risks remain explicit even when protected attributes are unavailable.
5. The notebook is explanatory; reusable code and tests remain the implementation source of truth.
