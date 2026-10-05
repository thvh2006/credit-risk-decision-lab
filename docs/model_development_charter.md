# Model development charter

## Decision and users

Estimate risk at application time to support a three-way underwriting policy:

- approve applications below a risk threshold;
- route an uncertainty band to trained manual reviewers;
- decline above the high-risk threshold only when legally permitted and accompanied
  by specific, accurate adverse-action reasons.

Primary users are a credit-risk policy owner, underwriting operations, independent
model validation, compliance, and portfolio monitoring. A model score is one input
to a controlled decision process, not an autonomous lending mandate.

## Target contract

The Kaggle `target` indicates whether a client defaulted on the credit case after
an unspecified period. Therefore this project estimates **the source target risk**.
It must not be represented as:

- a Basel one-year probability of default;
- an IFRS 9 lifetime or 12-month expected-credit-loss parameter;
- a contractual 30/60/90-days-past-due outcome;
- default risk for rejected applications whose counterfactual repayment is unknown.

If the performance window and default definition cannot be recovered from official
documentation, that ambiguity remains a material limitation and production blocker.

## Observation point

Scoring occurs at `date_decision`. Only information available as of that timestamp
is eligible. Historical child tables must be aggregated to one row per `case_id`
without using records that represent future information relative to the decision.
The competition host states that supplied inputs were collected as of application;
this does not eliminate the need to inspect date transforms and aggregation logic.

## Validation design

No random split is allowed for the headline result.

1. Development: earliest contiguous weeks.
2. Calibration: later contiguous weeks, untouched during model fitting.
3. Policy validation: later weeks used to choose operational thresholds.
4. Final OOT: latest contiguous weeks, locked until all choices are frozen.

Applicants sharing identifiers cannot be independently checked because a stable
customer identifier may be unavailable. Any evidence of repeated entities triggers
group-aware splitting or a limitation.

## Model stack

1. **Policy-naive benchmark:** portfolio base rate and simple rank baseline.
2. **Interpretable baseline:** regularised logistic model with documented missing
   handling and restrained feature set.
3. **Challenger:** XGBoost on application-time features and leakage-safe history
   aggregates.
4. **Stable challenger:** unstable features removed or constrained using only
   development/calibration evidence.
5. **Calibration layer:** sigmoid and isotonic candidates selected on the
   calibration window, never the OOT window.

Complexity wins only when the challenger improves OOT discrimination, calibration,
weekly stability, and operational policy value—not AUC alone.

## Primary metrics

- ROC-AUC and Gini for rank ordering.
- Average precision because defaults are imbalanced.
- Brier score, calibration intercept/slope, ECE, and reliability curves.
- Official weekly Gini stability score and worst-week performance.
- Approval rate, observed bad rate, captured defaults, exposure approved, and
  expected-loss scenarios under clearly labelled LGD assumptions.
- Group-level selection and error diagnostics where legitimate audit attributes
  exist; absence of protected-class data is not evidence of fairness.

## Prohibited shortcuts

- Random-only validation as the headline result.
- SMOTE before splitting or on the validation/test population.
- Accuracy as the main metric.
- Threshold selection on final OOT outcomes.
- Calling feature importance an explanation or causal driver.
- Fabricated interest income, LGD, recovery, capital, or regulatory PD.
- Training on protected attributes or obvious proxies without legal review.
- Inferring labels for rejected applicants and presenting them as ground truth.
