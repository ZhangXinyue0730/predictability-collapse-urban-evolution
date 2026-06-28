#!/usr/bin/env python3
"""Create a self-refreshing progress dashboard for the 79-city batch."""
from __future__ import annotations

import argparse
import csv
import html
import subprocess
import time
from datetime import datetime
from pathlib import Path


STATUS_CN = {
    "complete": "已完成",
    "complete_existing": "已完成",
    "failed": "失败",
}

STEP_CN = {
    "06_neighborhood_features.py": "邻里特征",
    "08_functional_features.py": "功能特征",
    "09.5_dynamic_builtup_extent.py": "动态建成区",
    "10_assemble_dataset.py": "组装训练数据",
    "11_train_and_ablate.py": "模型训练与消融实验",
    "12_shap_analysis.py": "整体 SHAP",
    "13_update_transition_stats.py": "分时期用地转移统计",
    "14_period_shap_analysis.py": "分时期 SHAP",
}


def read_status(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def batch_is_running() -> bool:
    result = subprocess.run(
        ["ps", "-axo", "command"],
        capture_output=True,
        text=True,
        check=False,
    )
    return "run_nonvacant_dynamic_t1_batch.py" in result.stdout


def current_step(log_path: Path) -> str:
    if not log_path.exists():
        return "准备中"
    latest = ""
    with log_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("RUN "):
                latest = line.removeprefix("RUN ").strip()
    return STEP_CN.get(latest, latest or "计算中")


def elapsed_seconds(started_at: str) -> str:
    if not started_at:
        return ""
    try:
        started = datetime.fromisoformat(started_at)
    except ValueError:
        return ""
    return str(int((datetime.now() - started).total_seconds()))


def build_rows(runs: Path) -> tuple[list[dict[str, str]], bool]:
    status_path = runs / "nonvacant_dynamic_t1_batch_status.csv"
    status_rows = read_status(status_path)
    by_city = {row["city_folder"]: row for row in status_rows}
    city_dirs = sorted(runs.glob("*全流程"))
    running = batch_is_running()
    next_index = len(status_rows) if running and len(status_rows) < len(city_dirs) else -1

    rows = []
    for index, city_dir in enumerate(city_dirs):
        raw = by_city.get(city_dir.name)
        if raw:
            status = STATUS_CN.get(raw["status"], raw["status"])
            step = "全部步骤完成" if status == "已完成" else raw.get("failed_step", "")
            duration = ""
            if raw.get("started_at") and raw.get("finished_at"):
                start = datetime.fromisoformat(raw["started_at"])
                finish = datetime.fromisoformat(raw["finished_at"])
                duration = str(int((finish - start).total_seconds()))
            finished_at = raw.get("finished_at", "")
        elif index == next_index:
            status = "运行中"
            log_path = runs / "_batch_logs_nonvacant_revision" / f"{city_dir.name}.log"
            step = current_step(log_path)
            started_at = ""
            if log_path.exists():
                started_at = datetime.fromtimestamp(log_path.stat().st_ctime).isoformat(
                    timespec="seconds"
                )
            duration = elapsed_seconds(started_at)
            finished_at = ""
        else:
            status = "待运行"
            step = ""
            duration = ""
            finished_at = ""

        rows.append({
            "序号": str(index + 1),
            "城市": city_dir.name.removesuffix("全流程"),
            "状态": status,
            "当前步骤": step,
            "耗时_秒": duration,
            "完成时间": finished_at,
        })
    return rows, running


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_html(path: Path, rows: list[dict[str, str]], running: bool) -> None:
    counts = {
        status: sum(row["状态"] == status for row in rows)
        for status in ("已完成", "运行中", "待运行", "失败")
    }
    percent = counts["已完成"] / len(rows) * 100
    body_rows = []
    for row in rows:
        status_class = {
            "已完成": "done",
            "运行中": "running",
            "待运行": "pending",
            "失败": "failed",
        }[row["状态"]]
        cells = "".join(f"<td>{html.escape(value)}</td>" for value in row.values())
        body_rows.append(f'<tr class="{status_class}">{cells}</tr>')

    dashboard = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="10">
  <title>79城市新口径实时进程</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", sans-serif;
      margin: 28px; color: #17202a; background: #f5f7f8; }}
    .summary {{ display: flex; gap: 14px; flex-wrap: wrap; margin: 18px 0; }}
    .card {{ background: white; padding: 12px 18px; border-radius: 10px;
      box-shadow: 0 2px 8px #0001; }}
    .bar {{ height: 18px; background: #dfe6e9; border-radius: 9px; overflow: hidden; }}
    .bar span {{ display: block; width: {percent:.2f}%; height: 100%; background: #138a72; }}
    table {{ width: 100%; border-collapse: collapse; background: white; margin-top: 18px; }}
    th, td {{ padding: 9px 12px; border-bottom: 1px solid #e8ecef; text-align: left; }}
    th {{ position: sticky; top: 0; background: #244f6e; color: white; }}
    tr.done td:nth-child(3) {{ color: #087f5b; font-weight: 700; }}
    tr.running {{ background: #fff3bf; }}
    tr.running td:nth-child(3) {{ color: #d9480f; font-weight: 700; }}
    tr.pending {{ color: #8a949b; }}
    tr.failed {{ background: #ffe3e3; color: #c92a2a; }}
    .note {{ color: #68737b; font-size: 14px; }}
  </style>
</head>
<body>
  <h1>79 城市新口径实时进程</h1>
  <div class="note">每 10 秒自动刷新 · 最后更新：
    {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ·
    批处理：{"运行中" if running else "已暂停或结束"}</div>
  <div class="summary">
    <div class="card"><b>已完成</b><br>{counts["已完成"]} / {len(rows)}</div>
    <div class="card"><b>运行中</b><br>{counts["运行中"]}</div>
    <div class="card"><b>待运行</b><br>{counts["待运行"]}</div>
    <div class="card"><b>失败</b><br>{counts["失败"]}</div>
    <div class="card"><b>总体进度</b><br>{percent:.1f}%</div>
  </div>
  <div class="bar"><span></span></div>
  <table>
    <thead><tr>{"".join(f"<th>{key}</th>" for key in rows[0])}</tr></thead>
    <tbody>{"".join(body_rows)}</tbody>
  </table>
</body>
</html>"""
    path.write_text(dashboard, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--interval", type=int, default=10)
    args = parser.parse_args()

    runs = args.root.resolve() / "city_extract_national" / "pipeline_runs_dynamic_t1"
    csv_path = runs / "79城市新口径实时运行进程.csv"
    html_path = runs / "79城市新口径实时运行进程.html"

    while True:
        rows, running = build_rows(runs)
        write_csv(csv_path, rows)
        write_html(html_path, rows, running)
        if not running:
            break
        time.sleep(args.interval)

    print(csv_path)
    print(html_path)


if __name__ == "__main__":
    main()
