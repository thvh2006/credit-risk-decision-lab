# Leakage register and feature policy

## Split-only fields

`WEEK_NUM`, `MONTH`, and `date_decision` define cohorts, chronology, drift, and
validation. They are prohibited as model predictors because the competition host
removed them from the hidden production-like test and warned against time-feature
metric exploitation.

## Identifier fields

`case_id`, row numbers, source-file partitions, and aggregation indexes are keys,
not risk predictors. File identity is forbidden because source files are themselves
partitioned by week and can leak time.

## Post-decision information

Every date-like field is checked relative to `date_decision`. Child-table aggregates
must use records available as of the observation point. Features whose semantics
refer to the performance outcome, future collection, or subsequent repayment are
excluded even if statistically powerful.

## Sensitive/protected and proxy-risk fields

Sex, nationality, marital/family status, disability/health, race/ethnicity/religion
where present, precise geography, and free text are not model inputs. Age and public-
assistance-like fields require jurisdiction-specific legal review and are excluded
from the main challenger. Permitted audit attributes are stored separately and never
exported in the scoring payload.

Proxy risk is assessed through feature definitions, missingness patterns, feature
importance, group distributions, and ablation—not through a claim that correlation
testing can prove a variable harmless.

## Missingness

Missingness can encode product eligibility, provider coverage, or prior lending
history. It is therefore treated as information to understand, not automatically
median-imputed. Each model documents whether it uses native missing handling,
explicit indicators, an “unknown” category, or a train-fitted imputer.

## Review artefact

The final feature inventory must contain:

| Field | Source table | Available at decision | Meaning | Missing treatment | Model/audit/excluded | Exclusion reason |
|---|---|---|---|---|---|---|

No model is considered reproducible until this inventory and its checksum are
generated from the same feature pipeline used for training.

