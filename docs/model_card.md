# Model card — depth-0 XGBoost challenger

## Intended decision and non-use

The model ranks application default risk to support a controlled approve / manual
review / decline simulation. It is a portfolio case study, not a deployable lending
system. It must not be used as a regulatory PD, for fully automated adverse action,
or in a jurisdiction without legal, fair-lending, privacy, validation, and model-risk
review.

## Training design

- **Fit:** deterministic 600,000-row sample from weeks 0–47.
- **Tuning:** all observations from weeks 48–54; AUC early stopping.
- **Calibration:** Platt calibration on weeks 55–68.
- **Policy selection:** weeks 69–77.
- **Final evaluation:** locked weeks 78–91.
- **Baseline:** L2 linear log-loss model on numeric features, median imputation and
  missingness indicators.
- **Challenger:** XGBoost histogram trees, depth 6, learning rate 0.035,
  column/row subsampling and regularisation; best iteration 1,194.

This sequence prevents the common leakage pattern where the same validation data is
used for early stopping, calibration, threshold selection, and final reporting.

## Results

| Model | Cohort | ROC AUC | Gini | Average precision | Brier | ECE | Observed rate | Mean predicted PD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Linear | Calibration | 0.6846 | 0.3693 | 0.1027 | 0.04210 | 0.00679 | 4.5135% | 4.5133% |
| XGBoost | Calibration | 0.7923 | 0.5847 | 0.1789 | 0.03997 | 0.00226 | 4.5135% | 4.5151% |
| Linear | Locked OOT | 0.7113 | 0.4227 | 0.0644 | 0.02080 | 0.02006 | 2.1192% | 4.1255% |
| XGBoost | Locked OOT | **0.8253** | **0.6507** | **0.1384** | **0.01973** | **0.01286** | 2.1192% | 3.4055% |

Average precision is about **6.5 times** the OOT base rate. The challenger improves
OOT AUC by 0.114 and average precision by 0.074 over the linear baseline.

![Model comparison](../reports/figures/oot_model_comparison.png)

## Stability interpretation

The competition-style OOT stability score is **0.6164**, with weekly mean Gini
0.6398, a non-negative weekly slope, and residual standard deviation 0.0468. Risk
ordering therefore survives the later cohorts well.

Calibration does not. The challenger overpredicts the aggregate OOT rate by 1.286
percentage points (3.406% versus 2.119%). This is consistent with calibration being
fitted in a high-default regime. It is not fixed using OOT outcomes because doing so
would invalidate the locked test. A production design would require delayed-label
recalibration, calibration-intercept monitoring, and fallback rules while labels
mature.

### Delayed-label recalibration backtest

To test that production design without altering the headline locked-OOT result, an
intercept-only correction is estimated on early OOT weeks 78–84 and evaluated on
strictly later weeks 85–91 (102,366 applications). It preserves every applicant's
risk ordering.

| Version | Evaluation AUC | Brier | ECE | Observed rate | Mean predicted PD |
|---|---:|---:|---:|---:|---:|
| Original calibration | 0.8335 | 0.01909 | 0.01352 | 2.057% | 3.409% |
| Intercept refresh | 0.8335 | **0.01880** | **0.00269** | 2.057% | 2.310% |

The correction reduces ECE by **80.1%** and leaves discrimination unchanged. This is
evidence that much of the observed error is a prevalence-level shift; it is not a
claim that all calibration drift is solved.

## Driver review

The leading global inputs are dominated by prior delinquency and repayment behaviour.
In a deterministic 10,000-application OOT sample, the largest positive local log-odds
contribution is aggregated into a privacy-safe reason-code report. Monthly annuity is
the top local reason for 22.9% of sampled applications; average tolerated DPD at
closure accounts for 12.9%, most-recent rejection date 9.6%, shared-mobile count 9.5%,
and payment count 9.2%.

![Recalibration and reason-code evidence](../reports/figures/recalibration_and_reasons.png)

These are **candidate internal reason codes**, not customer-facing adverse-action
notices. Direction, actionability, semantic accuracy, correlation, stability, and
legal suitability must be reviewed before communication. Global gain still must not
be substituted for a local explanation.

## Fairness boundary

Birth-date, marital-status, and education features are excluded from training. That
does not establish fairness: proxy effects may remain in behavioural and bureau
features. Group performance cannot be responsibly reported because the released data
does not provide a validated audit-label framework for protected classes. A real
deployment requires lawful collection of audit attributes, group-wise calibration,
error/approval analysis, intersectional sample checks, and remediation governance.

## Known risks and controls

| Risk | Evidence | Required control |
|---|---|---|
| Prior probability shift | OOT observed 2.12%, predicted 3.41% | calibration-gap trigger and recalibration |
| Provider coverage shift | 26,183 cases lack external static record | missingness and provider-availability monitoring |
| Proxy discrimination | behavioural/bureau fields can correlate with protected traits | legal review, group audit, ablations |
| Explanation mismatch | global gain is not a customer reason | constrained local reason-code pipeline |
| Target ambiguity | no public horizon/regulatory definition | no Basel/IFRS claim; validate outcome construction |
| Reject inference | outcomes are observed under historical approval policy | selection-bias analysis before expansion |

## Reproducibility

The random seed, split weeks, fit sample size, early-stopping iteration, fitted feature
manifest, and claim boundary are saved in `reports/run_metadata.json`. Tests exercise
the temporal split, stability metric, data contract, and monitoring triggers.
