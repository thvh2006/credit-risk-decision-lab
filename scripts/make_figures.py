"""Render aggregate, publication-safe figures from reproducible reports."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPORTS = Path("reports")
FIGURES = REPORTS / "figures"


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    weekly = pd.read_csv(REPORTS / "weekly_population.csv")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(weekly.WEEK_NUM, weekly.default_rate * 100, color="#1f4e79", linewidth=2)
    for left, right, label, color in (
        (0, 54, "Development", "#dbeafe"),
        (55, 68, "Calibration", "#dcfce7"),
        (69, 77, "Policy", "#fef3c7"),
        (78, 91, "Locked OOT", "#fee2e2"),
    ):
        ax.axvspan(left, right, alpha=0.45, color=color, label=label)
    ax.set(title="Default rate changes materially over time", xlabel="Cohort week", ylabel="Default rate (%)")
    ax.legend(ncol=4, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "weekly_default_rate.png", dpi=180)
    plt.close(fig)

    metrics = pd.read_csv(REPORTS / "model_metrics.csv")
    oot = metrics.query("partition == 'oot'").set_index("model")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    oot[["roc_auc", "average_precision"]].plot.bar(ax=axes[0], color=["#1f4e79", "#2a9d8f"])
    axes[0].set(title="Locked OOT discrimination", xlabel="", ylabel="Score", ylim=(0, 0.9))
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(["ROC AUC", "Average precision"], frameon=False)
    oot[["observed_default_rate", "mean_predicted_pd"]].mul(100).plot.bar(
        ax=axes[1], color=["#6b7280", "#e76f51"]
    )
    axes[1].set(title="OOT calibration level", xlabel="", ylabel="Rate (%)")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(["Observed", "Predicted"], frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "oot_model_comparison.png", dpi=180)
    plt.close(fig)

    policy = pd.read_csv(REPORTS / "oot_policy_validation.csv").query("lgd_assumption == 0.45")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(policy.oot_approval_rate * 100, policy.oot_observed_bad_rate * 100, "o-", color="#9b2226")
    for row in policy.itertuples():
        ax.annotate(
            f"target {row.target_policy_approval_rate:.0%}",
            (row.oot_approval_rate * 100, row.oot_observed_bad_rate * 100),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set(title="Frozen policy cut-offs on locked OOT", xlabel="Actual approval rate (%)", ylabel="Observed bad rate (%)")
    fig.tight_layout()
    fig.savefig(FIGURES / "oot_policy_frontier.png", dpi=180)
    plt.close(fig)

    recalibration = pd.read_csv(REPORTS / "recalibration_backtest.csv").set_index("version")
    reasons = pd.read_csv(REPORTS / "reason_code_summary.csv").head(8).sort_values(
        "share_as_top_reason"
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    calibration_plot = recalibration[
        ["observed_default_rate", "mean_predicted_pd"]
    ].mul(100)
    calibration_plot.plot.bar(ax=axes[0], color=["#6b7280", "#e76f51"])
    axes[0].set(
        title="Delayed-label calibration backtest (weeks 85–91)",
        xlabel="",
        ylabel="Rate (%)",
    )
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].set_xticklabels(["Original", "Intercept refresh"])
    axes[0].legend(["Observed", "Predicted"], frameon=False)
    axes[1].barh(
        reasons.feature,
        reasons.share_as_top_reason * 100,
        color="#1f4e79",
    )
    axes[1].set(
        title="Top local positive-risk reason in OOT sample",
        xlabel="Share of applications (%)",
        ylabel="",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "recalibration_and_reasons.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
