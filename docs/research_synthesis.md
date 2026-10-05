# Research synthesis and implementation consequences

This is a decision record: each source must change the analysis, model controls, or
claim boundary.

## 1. Time stability belongs in the objective

Home Credit's official metric computes weekly Gini, fits its time trend, penalises
negative slope, and penalises residual variability. The host describes stability as
important because a credit-scoring model may have a lifecycle of a year or more.

Source:
https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/overview/evaluation

**Implementation consequence:** validation is chronological; the repository
reproduces the official stability metric; model comparison includes worst-week and
slope diagnostics; `WEEK_NUM`, `MONTH`, and `date_decision` are split/index fields,
not production predictors.

## 2. Model risk is broader than predictive performance

The 2026 interagency US guidance supersedes SR 11-7 and emphasises risk-based model
risk management, clear intended use and limitations, conceptual soundness,
validation before use where possible, monitoring, outcome analysis, controls, and
effective challenge proportionate to model materiality.

Source:
https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm

**Implementation consequence:** the project ships a development charter, model
inventory entry, validation checklist, change log, use restrictions, monitoring
triggers, and challenger comparison. Passing CI is not called independent validation.

## 3. Creditworthiness is more than a score

EBA loan-origination guidance combines prudential and consumer-protection goals and
expects robust creditworthiness assessment, governance, data verification,
monitoring, and fair treatment through the credit lifecycle.

Source:
https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/credit-risk/guidelines-loan-origination-and-monitoring

**Implementation consequence:** policy outputs include affordability/data-quality
review flags and human review. A low predicted default risk cannot compensate for
missing required documentation or an unaffordable facility.

## 4. Discrimination and calibration answer different questions

Basel materials distinguish rank ordering and PD calibration, require representative
development data, review of inputs, known-bias assessment, human review, regular
validation, stability monitoring, and outcomes testing.

Source:
https://www.bis.org/committees/bcbs/basel-framework/standard/cre/36/inforce/2019-12-15/published/2019-12-15

**Implementation consequence:** AUC/Gini cannot approve a model alone. The project
reports Brier/ECE/reliability, risk-band observed-versus-predicted default, feature
and score drift, and outcome backtesting. It does not call the Kaggle target a
Basel-compliant PD because its horizon and default definition are incomplete.

## 5. Complex models do not relax adverse-action duties

The US CFPB states that ECOA/Regulation B requirements for specific and accurate
adverse-action reasons apply regardless of model complexity.

Source:
https://www.consumerfinance.gov/compliance/circulars/circular-2022-03-adverse-action-notification-requirements-in-connection-with-credit-decisions-based-on-complex-algorithms/

**Implementation consequence:** the challenger must map local contributions to a
controlled reason-code dictionary, test fidelity, and avoid vague explanations.
SHAP plots are development diagnostics; they are not automatically compliant notices.

## 6. Credit scoring is a high-risk AI use case in the EU

The EU AI Act classifies systems evaluating natural-person creditworthiness or
credit score as high-risk because they affect access to essential financial services
and can perpetuate discrimination.

Source:
https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689

**Implementation consequence:** the portfolio design includes data governance,
logging, human oversight, use restrictions, risk management, performance monitoring,
and group diagnostics. This document is not a conformity assessment.

## 7. Booked-loan labels create selection bias

Research on reject inference shows that repayment outcomes generally exist only for
accepted applicants, and that popular reject-inference heuristics rely on strong,
often untestable assumptions; no method is universally superior.

Sources:
- https://doi.org/10.1016/j.jbankfin.2003.10.010
- https://pmc.ncbi.nlm.nih.gov/articles/PMC9041715/

**Implementation consequence:** the model's empirical validity is limited to the
labelled/financed population represented by the source. The project discusses reject
inference but does not fabricate outcomes for declined applicants. A production
lender would need controlled exploration, external outcomes, or a defensible
selection model with independent validation.

## 8. Fairness cannot be inferred from missing demographics

Fair-lending review considers disparate treatment and facially neutral policies that
may have disparate effects on protected groups. Group testing needs legitimate,
governed audit data and legal interpretation.

Sources:
- https://www.federalreserve.gov/frrs/regulations/policy-statement-on-discrimination-in-lending.htm
- https://www.fdic.gov/system/files/2024-06/iv-1-1.pdf

**Implementation consequence:** protected or sensitive fields are excluded from
model training by default and retained only in a restricted audit layer when the
dataset permits. Reported group gaps are diagnostics, not legal conclusions. If the
source lacks a protected attribute, the report says “not measurable,” never “fair.”

