from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pandas as pd


SCRIPT = Path(os.environ.get("MULTIMODEL_SCRIPT", "scripts/analysis/multimodel_auc_robustness.py")).resolve()


def load_module():
    spec = importlib.util.spec_from_file_location("multimodel_auc_robustness", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    m = load_module()
    detail = pd.read_csv(m.OUT_DETAIL)

    # The pipeline master table used "1984->1990", while the robustness script
    # used "1984-1990". Normalize to one label system before summarizing.
    detail["period"] = detail["period"].astype(str).str.replace("->", "-", regex=False)
    detail["period_short"] = detail["period"].map(m.PERIOD_SHORT)
    detail.to_csv(m.OUT_DETAIL, index=False, encoding="utf-8-sig")

    table_a1 = m.summarize_for_table_a1(detail)
    table_a1.to_csv(m.OUT_TABLE_CSV, index=False, encoding="utf-8-sig")
    m.plot_figure_a1(detail)

    print("Corrected outputs:")
    print(m.OUT_DETAIL)
    print(m.OUT_TABLE_CSV)
    print(m.OUT_FIG)
    print()
    print(table_a1.to_string(index=False))


if __name__ == "__main__":
    main()
