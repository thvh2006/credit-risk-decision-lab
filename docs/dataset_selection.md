# Dataset selection record

## Decision

Use the **Home Credit — Credit Risk Model Stability (2024)** competition data as
an independent credit-risk case study. Do not imply that these applicants belong
to NovaPay, ShopFlow, or any other portfolio company.

## Candidate comparison

| Candidate | Strength | Material limitation | Decision |
|---|---|---|---|
| Home Credit Stability 2024 | ~1.5M applications, weekly cohorts, rich relational history, explicit stability objective | 26.77 GB; transformed features; target window is not fully disclosed | **Selected** |
| Home Credit Default Risk 2018 | 307k labelled applications and familiar relational schema | no application timestamp for true OOT validation | Rejected as primary |
| Freddie/Fannie performance data | real mortgages, performance history and true vintage analysis | registration, specialised mortgage context, much larger domain scope | Future extension |
| UCI German / South German Credit | open and easy to reproduce | only ~1,000 rows; too small for stability and policy work | Rejected |
| LendingClub public extracts | loan terms and repayment outcomes | provenance/version/licensing fragmentation and booked-loan selection | Rejected |

## Why the 2024 source fits the learning objective

The official competition evaluates a Gini score inside each `WEEK_NUM`, then
penalises a negative performance slope and residual variability. That makes time
stability a first-class modelling constraint rather than a paragraph added after a
random train/test split.

Source:
https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability

## Access and licence boundary

The files are subject to Kaggle competition rules. Raw or prepared data will not be
committed or redistributed. Reproduction requires the user to join the competition
and obtain the data through Kaggle. Only derived aggregate metrics, model artefacts
permitted by the rules, code, and figures belong in the public repository.

