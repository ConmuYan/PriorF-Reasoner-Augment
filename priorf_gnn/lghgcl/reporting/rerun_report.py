"""Aggregate repaired-code rerun outputs and export figures."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

METRICS = ("AUPRC", "AUROC", "F1-Macro", "G-means")
MAIN_BRANCH_ORDER = ("full", "scre", "raw_gnn", "no_sdcl", "mlp_only", "gnn_only")
MAIN_BRANCH_LABELS = {
    "full": "PriorF-GNN (full)",
    "scre": "SCRE",
    "raw_gnn": "Raw GNN",
    "no_sdcl": "No SDCL",
    "mlp_only": "MLP only",
    "gnn_only": "GNN only",
}
SCARCITY_BRANCH_ORDER = ("full", "scre", "no_sdcl")
SCARCITY_RATIO_ORDER = (40, 20, 10, 5, 1)
SCHEDULE_ORDER = ("cosine", "constant", "increasing")
RESERVED_DIRS = {"_smoke", "aggregate", "queues", "logs", "cross_dataset"}
BRANCH_COLORS = {
    "full": "#1b9e77",
    "scre": "#d95f02",
    "raw_gnn": "#7570b3",
    "no_sdcl": "#e7298a",
    "mlp_only": "#66a61e",
    "gnn_only": "#e6ab02",
    "cosine": "#1b9e77",
    "constant": "#d95f02",
    "increasing": "#7570b3",
}


@dataclass
class RunRecord:
    dataset: str
    family: str
    variant: str
    experiment_name: str
    seed: int
    scarcity_ratio: int | None
    schedule: str | None
    run_dir: str
    metrics: dict[str, float]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _metric_value(metrics: dict[str, float], key: str) -> float:
    value = metrics.get(key, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_branch(name: str) -> tuple[str, str, int | None, str | None] | None:
    if name.startswith("main_"):
        return ("main", name.removeprefix("main_"), None, None)
    scarcity_match = re.match(r"^scarcity_(?P<variant>.+)_(?P<label>\d+)$", name)
    if scarcity_match:
        return ("scarcity", scarcity_match.group("variant"), int(scarcity_match.group("label")), None)
    if name.startswith("schedule_"):
        return ("schedule", "full", None, name.removeprefix("schedule_"))
    return None


def _metric_slug(metric: str) -> str:
    return metric.lower().replace("-", "_")


def _dataset_name(root_dir: Path, dataset: str | None) -> str:
    return (dataset or root_dir.name).lower()


def scan_dataset_runs(root_dir: Path, dataset: str | None = None) -> list[RunRecord]:
    dataset_name = _dataset_name(root_dir, dataset)
    records: list[RunRecord] = []
    for branch_dir in sorted(root_dir.iterdir()):
        if not branch_dir.is_dir():
            continue
        if branch_dir.name in RESERVED_DIRS or branch_dir.name.startswith("_"):
            continue
        parsed = _parse_branch(branch_dir.name)
        if parsed is None:
            continue
        family, variant, scarcity_ratio, schedule = parsed
        for seed_dir in sorted(branch_dir.glob("seed_*")):
            if not seed_dir.is_dir():
                continue
            try:
                seed = int(seed_dir.name.removeprefix("seed_"))
            except ValueError:
                continue
            metrics_path = seed_dir / "best_test_metrics.json"
            if not metrics_path.exists():
                continue
            metrics_raw = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics = {key: _metric_value(metrics_raw, key) for key in (*METRICS, "Threshold")}
            records.append(
                RunRecord(
                    dataset=dataset_name,
                    family=family,
                    variant=variant,
                    experiment_name=branch_dir.name,
                    seed=seed,
                    scarcity_ratio=scarcity_ratio,
                    schedule=schedule,
                    run_dir=str(seed_dir),
                    metrics=metrics,
                )
            )
    return records


def _records_to_rows(records: list[RunRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = {
            "dataset": record.dataset,
            "family": record.family,
            "variant": record.variant,
            "experiment_name": record.experiment_name,
            "seed": record.seed,
            "scarcity_ratio": "" if record.scarcity_ratio is None else record.scarcity_ratio,
            "schedule": record.schedule or "",
            "run_dir": record.run_dir,
        }
        row.update(record.metrics)
        rows.append(row)
    return rows


def _summarize_records(records: list[RunRecord]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int | None, str | None], list[RunRecord]] = {}
    for record in records:
        key = (record.dataset, record.family, record.variant, record.scarcity_ratio, record.schedule)
        grouped.setdefault(key, []).append(record)

    summary_rows: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: (item[1], item[2], item[3] or -1, item[4] or "")):
        bucket = grouped[key]
        row = {
            "dataset": key[0],
            "family": key[1],
            "variant": key[2],
            "scarcity_ratio": "" if key[3] is None else key[3],
            "schedule": key[4] or "",
            "num_seeds": len(bucket),
        }
        for metric in METRICS:
            values = [_metric_value(record.metrics, metric) for record in bucket]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = _safe_std(values)
        summary_rows.append(row)
    return summary_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_figure(fig: plt.Figure, output_prefix: Path) -> None:
    _ensure_dir(output_prefix.parent)
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_main_figures(summary_rows: list[dict[str, Any]], output_dir: Path, dataset: str) -> None:
    main_rows = [row for row in summary_rows if row["family"] == "main"]
    if not main_rows:
        return
    rows_by_variant = {row["variant"]: row for row in main_rows}
    variants = [name for name in MAIN_BRANCH_ORDER if name in rows_by_variant]
    labels = [MAIN_BRANCH_LABELS[name] for name in variants]
    x = np.arange(len(variants))

    for metric in METRICS:
        means = [rows_by_variant[name][f"{metric}_mean"] for name in variants]
        stds = [rows_by_variant[name][f"{metric}_std"] for name in variants]
        colors = [BRANCH_COLORS.get(name, "#4c78a8") for name in variants]
        fig, ax = plt.subplots(figsize=(11, 5))
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"{dataset} main ablations: {metric}")
        for bar, mean in zip(bars, means, strict=False):
            ax.text(bar.get_x() + bar.get_width() / 2, mean, f"{mean:.4f}", ha="center", va="bottom", fontsize=8)
        _save_figure(fig, output_dir / f"{dataset}_main_{_metric_slug(metric)}")


def _plot_scarcity_figures(summary_rows: list[dict[str, Any]], output_dir: Path, dataset: str) -> None:
    scarcity_rows = [row for row in summary_rows if row["family"] == "scarcity"]
    if not scarcity_rows:
        return
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(10, 5))
        for variant in SCARCITY_BRANCH_ORDER:
            branch_rows = [row for row in scarcity_rows if row["variant"] == variant]
            if not branch_rows:
                continue
            row_by_ratio = {int(row["scarcity_ratio"]): row for row in branch_rows if row["scarcity_ratio"] != ""}
            xs = [ratio for ratio in SCARCITY_RATIO_ORDER if ratio in row_by_ratio]
            means = [row_by_ratio[ratio][f"{metric}_mean"] for ratio in xs]
            stds = [row_by_ratio[ratio][f"{metric}_std"] for ratio in xs]
            ax.errorbar(
                xs,
                means,
                yerr=stds,
                marker="o",
                linewidth=2,
                capsize=4,
                color=BRANCH_COLORS.get(variant, "#4c78a8"),
                label=MAIN_BRANCH_LABELS.get(variant, variant),
            )
        ax.set_xticks(list(SCARCITY_RATIO_ORDER), [f"{ratio}%" for ratio in SCARCITY_RATIO_ORDER])
        ax.invert_xaxis()
        ax.set_ylabel(metric)
        ax.set_xlabel("Train label ratio")
        ax.set_title(f"{dataset} scarcity: {metric}")
        ax.legend()
        _save_figure(fig, output_dir / f"{dataset}_scarcity_{_metric_slug(metric)}")


def _plot_stability_figures(records: list[RunRecord], output_dir: Path, dataset: str) -> None:
    main_records = [record for record in records if record.family == "main"]
    if not main_records:
        return
    for metric in ("AUPRC", "AUROC"):
        fig, ax = plt.subplots(figsize=(11, 5))
        variants = [name for name in MAIN_BRANCH_ORDER if any(record.variant == name for record in main_records)]
        for index, variant in enumerate(variants):
            ys = [_metric_value(record.metrics, metric) for record in main_records if record.variant == variant]
            if not ys:
                continue
            xs = np.linspace(index - 0.12, index + 0.12, len(ys))
            ax.scatter(xs, ys, color=BRANCH_COLORS.get(variant, "#4c78a8"), alpha=0.9, s=42)
        ax.set_xticks(np.arange(len(variants)), [MAIN_BRANCH_LABELS[name] for name in variants], rotation=20, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"{dataset} seed stability: {metric}")
        _save_figure(fig, output_dir / f"{dataset}_seed_stability_{_metric_slug(metric)}")


def _plot_schedule_figures(summary_rows: list[dict[str, Any]], output_dir: Path, dataset: str) -> None:
    schedule_rows = [row for row in summary_rows if row["family"] == "schedule"]
    if not schedule_rows:
        return
    rows_by_schedule = {row["schedule"]: row for row in schedule_rows}
    schedules = [name for name in SCHEDULE_ORDER if name in rows_by_schedule]
    x = np.arange(len(schedules))
    for metric in METRICS:
        means = [rows_by_schedule[name][f"{metric}_mean"] for name in schedules]
        stds = [rows_by_schedule[name][f"{metric}_std"] for name in schedules]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x, means, yerr=stds, capsize=4, color=[BRANCH_COLORS.get(name, "#4c78a8") for name in schedules])
        ax.set_xticks(x, schedules)
        ax.set_ylabel(metric)
        ax.set_title(f"{dataset} schedule comparison: {metric}")
        _save_figure(fig, output_dir / f"{dataset}_schedule_{_metric_slug(metric)}")


def write_dataset_report(root_dir: Path | str, dataset: str | None = None) -> dict[str, Any]:
    root = Path(root_dir)
    dataset_name = _dataset_name(root, dataset)
    records = scan_dataset_runs(root, dataset_name)
    if not records:
        raise ValueError(f"No completed runs found under {root}")

    aggregate_dir = root / "aggregate"
    figure_root = aggregate_dir / "paper_figures"
    for folder in ("main", "scarcity", "stability", "schedule"):
        _ensure_dir(figure_root / folder)

    per_seed_rows = _records_to_rows(records)
    summary_rows = _summarize_records(records)
    main_summary = [row for row in summary_rows if row["family"] == "main"]
    scarcity_summary = [row for row in summary_rows if row["family"] == "scarcity"]
    schedule_summary = [row for row in summary_rows if row["family"] == "schedule"]

    _write_csv(aggregate_dir / "per_seed_metrics.csv", per_seed_rows)
    _write_json(aggregate_dir / "per_seed_metrics.json", per_seed_rows)
    _write_csv(aggregate_dir / "main_5seed_summary.csv", main_summary)
    _write_json(aggregate_dir / "main_5seed_summary.json", main_summary)
    _write_csv(aggregate_dir / "scarcity_5seed_summary.csv", scarcity_summary)
    _write_json(aggregate_dir / "scarcity_5seed_summary.json", scarcity_summary)
    if schedule_summary:
        _write_csv(aggregate_dir / "schedule_5seed_summary.csv", schedule_summary)
        _write_json(aggregate_dir / "schedule_5seed_summary.json", schedule_summary)

    _plot_main_figures(summary_rows, figure_root / "main", dataset_name)
    _plot_scarcity_figures(summary_rows, figure_root / "scarcity", dataset_name)
    _plot_stability_figures(records, figure_root / "stability", dataset_name)
    _plot_schedule_figures(summary_rows, figure_root / "schedule", dataset_name)

    report = {
        "dataset": dataset_name,
        "root_dir": str(root),
        "num_records": len(records),
        "num_main_rows": len(main_summary),
        "num_scarcity_rows": len(scarcity_summary),
        "num_schedule_rows": len(schedule_summary),
    }
    _write_json(aggregate_dir / "report_summary.json", report)
    return report


def _read_dataset_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_cross_dataset_metrics(all_main_rows: list[dict[str, Any]], output_dir: Path) -> None:
    full_rows = [row for row in all_main_rows if row["variant"] == "full"]
    if not full_rows:
        return
    x = np.arange(len(METRICS))
    width = 0.35 if len(full_rows) > 1 else 0.6
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, row in enumerate(full_rows):
        means = [row[f"{metric}_mean"] for metric in METRICS]
        stds = [row[f"{metric}_std"] for metric in METRICS]
        offset = (idx - (len(full_rows) - 1) / 2) * width
        ax.bar(x + offset, means, width=width, yerr=stds, capsize=4, label=row["dataset"])
    ax.set_xticks(x, METRICS, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("PriorF-GNN full model across datasets")
    ax.legend()
    _save_figure(fig, output_dir / "fullmodel_cross_dataset_metrics")


def _plot_cross_dataset_scarcity(all_scarcity_rows: list[dict[str, Any]], output_dir: Path) -> None:
    rows = [row for row in all_scarcity_rows if row["variant"] == "full"]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    datasets = sorted({row["dataset"] for row in rows})
    for dataset_name in datasets:
        row_by_ratio = {
            int(row["scarcity_ratio"]): row
            for row in rows
            if row["dataset"] == dataset_name and row["scarcity_ratio"] != ""
        }
        xs = [ratio for ratio in SCARCITY_RATIO_ORDER if ratio in row_by_ratio]
        ys = [row_by_ratio[ratio]["AUPRC_mean"] for ratio in xs]
        yerr = [row_by_ratio[ratio]["AUPRC_std"] for ratio in xs]
        ax.errorbar(xs, ys, yerr=yerr, marker="o", linewidth=2, capsize=4, label=dataset_name)
    ax.set_xticks(list(SCARCITY_RATIO_ORDER), [f"{ratio}%" for ratio in SCARCITY_RATIO_ORDER])
    ax.invert_xaxis()
    ax.set_xlabel("Train label ratio")
    ax.set_ylabel("AUPRC")
    ax.set_title("PriorF-GNN full-model scarcity trend")
    ax.legend()
    _save_figure(fig, output_dir / "fullmodel_cross_dataset_scarcity")


def write_cross_dataset_report(root_dir: Path | str) -> dict[str, Any]:
    root = Path(root_dir)
    dataset_dirs = [path for path in (root / name for name in ("amazon", "yelpchi")) if path.exists() and path.is_dir()]
    if not dataset_dirs:
        raise ValueError(f"No dataset directories found under {root}")

    cross_dir = root / "cross_dataset"
    figure_dir = cross_dir / "final_figures"
    _ensure_dir(figure_dir)

    all_main_rows: list[dict[str, Any]] = []
    all_scarcity_rows: list[dict[str, Any]] = []
    all_schedule_rows: list[dict[str, Any]] = []
    datasets: list[str] = []

    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name.lower()
        datasets.append(dataset_name)
        aggregate_dir = dataset_dir / "aggregate"
        if not aggregate_dir.exists():
            write_dataset_report(dataset_dir, dataset_name)
        all_main_rows.extend(_read_dataset_summary(aggregate_dir / "main_5seed_summary.json"))
        all_scarcity_rows.extend(_read_dataset_summary(aggregate_dir / "scarcity_5seed_summary.json"))
        all_schedule_rows.extend(_read_dataset_summary(aggregate_dir / "schedule_5seed_summary.json"))

    _write_csv(cross_dir / "final_main_table.csv", all_main_rows)
    _write_csv(cross_dir / "final_scarcity_table.csv", all_scarcity_rows)
    if all_schedule_rows:
        _write_csv(cross_dir / "final_schedule_table.csv", all_schedule_rows)

    _plot_cross_dataset_metrics(all_main_rows, figure_dir)
    _plot_cross_dataset_scarcity(all_scarcity_rows, figure_dir)

    report = {
        "root_dir": str(root),
        "datasets": sorted(datasets),
        "num_main_rows": len(all_main_rows),
        "num_scarcity_rows": len(all_scarcity_rows),
        "num_schedule_rows": len(all_schedule_rows),
    }
    _write_json(cross_dir / "report_summary.json", report)
    return report
