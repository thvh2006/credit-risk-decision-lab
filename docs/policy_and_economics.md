# Underwriting policy and economic assumptions

## Three-way policy

The final score is converted into actions only after calibration and OOT validation:

- **Approve:** risk below the lower threshold and all non-model policy gates pass.
- **Manual review:** uncertainty band, missing critical evidence, affordability
  concern, out-of-distribution score, or controlled exception.
- **Decline:** risk above the upper threshold, with specific accurate reasons and an
  appeal/reconsideration route where required.

Thresholds are selected on the policy-validation window and frozen before final OOT.

## What can be measured from the source

- application volume and default target rate;
- risk ranking and calibration;
- approved count and observed bad rate under simulated score cutoffs;
- requested/credit amount approved when a reliable exposure field is identified;
- default capture and review workload;
- time and group stability.

## What requires assumptions

Expected loss normally combines PD, exposure at default, and LGD. This source does
not provide a validated regulatory PD horizon, realised EAD, recoveries, collections
cost, funding cost, pricing, or contribution margin.

The policy table may show sensitivity scenarios such as:

```text
scenario expected loss = model risk × approved amount proxy × assumed LGD
```

Every assumed LGD is visible in the output and varied over a range. It is not an
estimate. No “expected profit” is reported unless all income and cost inputs come
from a documented source.

## Decision frontier

At minimum compare:

1. growth policy — higher approval, higher expected loss;
2. balanced policy — moderate approval plus review band;
3. conservative policy — lower approval and loss;
4. baseline scorecard versus calibrated challenger at the same approval rate;
5. all-period metric versus locked OOT performance.

The recommendation must state who gains access, who is declined, default capture,
operational review volume, scenario loss, group gaps, uncertainty, and conditions
that would reverse the decision.

