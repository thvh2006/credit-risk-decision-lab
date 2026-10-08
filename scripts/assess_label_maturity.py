"""Write an explicit label-maturity ambiguity diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.label_maturity import maturity_diagnostic

REPORTS = Path("reports")


def main() -> None:
    result = maturity_diagnostic(pd.read_csv(REPORTS / "weekly_population.csv"))
    (REPORTS / "label_maturity_diagnostic.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
