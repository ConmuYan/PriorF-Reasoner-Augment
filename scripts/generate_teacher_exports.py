"""Generate validated teacher export artifacts for Amazon and YelpChi.

Pipeline:
  1. Load canonical .mat (built by scripts/legacy_mat_to_canonical.py).
  2. Run LGHGCLNetV2 inference → TeacherExportRecords per population.
  3. Build + write a DataManifest with all three populations.
  4. Run teacher_baseline_gate on validation → TeacherBaselineReport.
  5. Call write_teacher_export_artifact for each population under
     outputs/gated/teacher_exports/<dataset>/<population>.
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parents[1]
PRIORF_GNN_ROOT = REPO_ROOT / "priorf_gnn"
for import_root in (REPO_ROOT, PRIORF_GNN_ROOT):
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)

from graph_data.manifests import (
    DataArtifact,
    PopulationMetadata,
    build_data_manifest,
    write_data_manifest,
)
from graph_data.mat_loader import load_standard_mat
from graph_data.validators import compute_node_ids_hash
from priorf_teacher.export_pipeline import write_teacher_export_artifact
from priorf_teacher.inference import run_teacher_inference
from priorf_teacher.schema import (
    DatasetName,
    GraphRegime,
    MetricName,
    PopulationName,
    TeacherExportManifest,
    TeacherProvenance,
)
from priorf_teacher.teacher_baseline_gate import run_teacher_baseline_gate

DATASETS = {
    "amazon": {
        "legacy_mat": "assets/data/Amazon.mat",
        "canonical_mat": "assets/data/Amazon_canonical.mat",
        "checkpoint": "assets/teacher/amazon/best_model.pt",
        "model_summary": "assets/teacher/amazon/model_summary.json",
        "dataset_enum": DatasetName.AMAZON,
    },
    "yelpchi": {
        "legacy_mat": "assets/data/YelpChi.mat",
        "canonical_mat": "assets/data/YelpChi_canonical.mat",
        "checkpoint": "assets/teacher/yelpchi/best_model.pt",
        "model_summary": "assets/teacher/yelpchi/model_summary.json",
        "dataset_enum": DatasetName.YELPCHI,
    },
}

ZERO_SHA = "0" * 40
GIT_SHA_PLACEHOLDER = "0" * 40
GRAPH_REGIME = GraphRegime.TRANSDUCTIVE_STANDARD
GRAPH_REGIME_STR = "transductive_standard"


def _file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _population_metadatas(
    canonical_mat_path: str,
) -> dict[str, PopulationMetadata]:
    raw = loadmat(canonical_mat_path, squeeze_me=True)
    split_vector = np.asarray(raw["split_vector"]).astype(str)
    node_ids = np.arange(len(split_vector), dtype=np.int64)

    pops: dict[str, PopulationMetadata] = {}
    for pop_name in ("train", "validation", "final_test"):
        mask = split_vector == pop_name
        if not mask.any():
            continue
        ids = node_ids[mask]
        pops[pop_name] = PopulationMetadata(
            population_name=pop_name,
            split_values=(pop_name,),
            node_ids_hash=compute_node_ids_hash(ids),
            contains_tuning_rows=(pop_name == "validation"),
            contains_final_test_rows=(pop_name == "final_test"),
        )
    return pops


def generate_for_dataset(
    dataset_key: str,
    base_output_dir: Path,
    device: str = "cuda",
) -> None:
    cfg = DATASETS[dataset_key]
    print(f"\n=== {dataset_key.upper()} ===")

    summary = json.loads(Path(cfg["model_summary"]).read_text())
    model_hyperparams = summary["config"]["model"]

    # 1. Run inference → per-population records.
    print("  [1/5] Running teacher inference...")
    pop_records = run_teacher_inference(
        dataset_name=cfg["dataset_enum"],
        legacy_mat_path=cfg["legacy_mat"],
        canonical_mat_path=cfg["canonical_mat"],
        checkpoint_path=cfg["checkpoint"],
        model_hyperparams=model_hyperparams,
        teacher_model_name="LGHGCLNetV2",
        teacher_checkpoint=cfg["checkpoint"],
        graph_regime=GRAPH_REGIME,
        device=device,
    )
    for pop_str, recs in pop_records.items():
        print(f"       {pop_str}: {len(recs)} records")

    # 2. Build + write DataManifest.
    print("  [2/5] Building data manifest...")
    canonical_data = load_standard_mat(cfg["canonical_mat"], dataset_name=dataset_key)
    source_sha256 = _file_sha256(cfg["canonical_mat"])
    populations_map = _population_metadatas(cfg["canonical_mat"])
    artifacts = (
        DataArtifact(
            kind="source_mat",
            path=str(Path(cfg["canonical_mat"]).resolve()),
            sha256=source_sha256,
        ),
    )
    manifest_dir = base_output_dir / "manifests" / dataset_key
    manifest_dir.mkdir(parents=True, exist_ok=True)
    data_manifest = build_data_manifest(
        data=canonical_data,
        graph_regime=GRAPH_REGIME_STR,  # DataManifest uses Literal strings
        populations=tuple(populations_map.values()),
        artifacts=artifacts,
    )
    data_manifest_path = manifest_dir / "data_manifest.json"
    write_data_manifest(data_manifest, data_manifest_path)
    data_manifest_sha256 = _file_sha256(data_manifest_path)
    print(f"       manifest: {data_manifest_path}")

    # 3. Run baseline gate on validation.
    print("  [3/5] Running teacher baseline gate on validation...")
    checkpoint_sha256 = _file_sha256(cfg["checkpoint"])
    val_records = pop_records.get("validation", [])
    if not val_records:
        raise RuntimeError("no validation records; cannot run baseline gate")
    val_labels = tuple(int(r.ground_truth_label) for r in val_records)
    val_probs = tuple(float(r.teacher_prob) for r in val_records)
    report_path = manifest_dir / "validation_baseline_report.json"
    report = run_teacher_baseline_gate(
        data_manifest_path=str(data_manifest_path),
        teacher_checkpoint_path=cfg["checkpoint"],
        teacher_model_name="LGHGCLNetV2",
        teacher_checkpoint_sha256=checkpoint_sha256,
        dataset_name=cfg["dataset_enum"],
        graph_regime=GRAPH_REGIME,
        # NOTE (Task 2 runtime acceptance, audit trail): metric_name / metric_threshold
        # below are operator hard-coded defaults. They are NOT prescribed by any
        # source-of-truth document (docs/010, docs/020 §10, docs/030 §Task 2
        # Acceptance, docs/050). In particular, metric_threshold=0.80 is a single
        # dataset-agnostic value and has no recorded calibration rationale.
        # On YelpChi this threshold currently blocks artifact production at
        # validation AUROC=0.7518 (see status_package/general.md §"Task 2
        # 运行时验收"). A diagnostic under scripts/verify_teacher_metric_consistency.py
        # exists to recompute (AUROC, AUPRC_trapezoidal, AUPRC_step) before any
        # threshold / metric change is proposed. Do NOT edit the two lines below
        # without (a) a verification-run log, and (b) a per-dataset rationale
        # committed to status_package/general.md.
        metric_name=MetricName.AUROC,
        metric_threshold=0.80,
        population_name=PopulationName.VALIDATION,
        validation_ground_truth_label=val_labels,
        validation_teacher_prob=val_probs,
        code_git_sha=GIT_SHA_PLACEHOLDER,
        report_path=str(report_path),
    )
    print(f"       AUROC={report.metric_value:.4f} passed={report.passed}")
    if not report.passed:
        raise RuntimeError(f"baseline gate failed: AUROC={report.metric_value}")

    # 4. Write teacher export artifact per population.
    print("  [4/5] Writing teacher export artifacts...")
    for pop_str, recs in pop_records.items():
        pop_meta = populations_map[pop_str]
        pop_enum = recs[0].population_name
        provenance = TeacherProvenance(
            code_git_sha=GIT_SHA_PLACEHOLDER,
            teacher_checkpoint_path=cfg["checkpoint"],
            teacher_checkpoint_sha256=checkpoint_sha256,
            data_manifest_path=str(data_manifest_path),
            data_manifest_sha256=data_manifest_sha256,
            export_timestamp_utc=datetime.now(timezone.utc),
            random_seed=717,
            graph_regime=GRAPH_REGIME,
        )
        export_manifest = TeacherExportManifest(
            dataset_name=cfg["dataset_enum"],
            population_name=pop_enum,
            graph_regime=GRAPH_REGIME,
            row_count=len(recs),
            node_ids_hash=pop_meta.node_ids_hash,
            split_values=(pop_str,),
            contains_tuning_rows=pop_meta.contains_tuning_rows,
            contains_final_test_rows=pop_meta.contains_final_test_rows,
            provenance=provenance,
        )
        output_dir = (
            base_output_dir
            / "outputs" / "gated" / "teacher_exports"
            / cfg["dataset_enum"].value
            / pop_enum.value
        )
        artifact_path, manifest_out_path = write_teacher_export_artifact(
            data_manifest_path=str(data_manifest_path),
            teacher_baseline_report_path=str(report_path),
            output_dir=output_dir,
            export_manifest=export_manifest,
            records=tuple(recs),
        )
        print(f"       {pop_str}: {artifact_path}")

    print("  [5/5] Done.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS.keys()) + ["all"], default="all")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        generate_for_dataset(ds, output_dir, device=args.device)


if __name__ == "__main__":
    main()
