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
import subprocess
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

# Files that MUST match their committed HEAD content for the embedded
# TeacherProvenance.code_git_sha to be a truthful audit anchor. Kept
# deliberately small: only the teacher inference bridge, the canonical-mat
# builder, this driver script, and the diagnostic. Anything outside this set
# (unrelated tracked files) is allowed to be dirty without blocking the run.
BRIDGE_FILES: tuple[str, ...] = (
    "priorf_teacher/inference.py",
    "scripts/legacy_mat_to_canonical.py",
    "scripts/generate_teacher_exports.py",
    "scripts/verify_teacher_metric_consistency.py",
)

GRAPH_REGIME = GraphRegime.TRANSDUCTIVE_STANDARD
GRAPH_REGIME_STR = "transductive_standard"


def _file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head_sha_or_fail(expected_clean_files: tuple[str, ...]) -> str:
    """Return a 40-char hex git HEAD SHA or raise.

    Fails closed (RuntimeError) if any of:
      * git is unavailable or returns a non-zero exit status,
      * HEAD SHA is not a 40-character lowercase hex string,
      * any of ``expected_clean_files`` has an uncommitted diff relative to
        HEAD (tracked modification, staged or unstaged, including rename /
        delete / untracked-inside-tracked-dir).

    The cleanliness check is intentionally narrow: it only asserts the
    bridge files listed by the caller, not the full working tree. Files
    outside ``expected_clean_files`` (e.g. IDE state, unrelated WIP) may
    be dirty without blocking the run, because they cannot change the
    behaviour of the code path that produces teacher artifacts.
    """
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.PIPE,
        ).decode("ascii").strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "cannot read git HEAD for TeacherProvenance.code_git_sha; "
            "refusing to embed a zero-SHA placeholder"
        ) from exc

    if len(head) != 40 or any(c not in "0123456789abcdef" for c in head):
        raise RuntimeError(
            f"git HEAD is not a 40-char lowercase hex SHA: {head!r}"
        )

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--", *expected_clean_files],
            cwd=str(REPO_ROOT),
            stderr=subprocess.PIPE,
        ).decode("utf-8")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "cannot run `git status --porcelain` for bridge cleanliness check"
        ) from exc

    if status.strip():
        raise RuntimeError(
            "bridge tracked files are dirty relative to HEAD; refusing to "
            "embed HEAD SHA in TeacherProvenance.code_git_sha.\n"
            f"offending `git status --porcelain` output:\n{status}"
        )

    return head


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
    code_git_sha: str | None = None,
) -> None:
    if code_git_sha is None or len(code_git_sha) != 40:
        raise RuntimeError(
            "generate_for_dataset requires a 40-char code_git_sha derived "
            "from _git_head_sha_or_fail; zero-SHA placeholders are not "
            "permitted after Task 2 runtime-acceptance commit."
        )
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
        # Acceptance, docs/050). metric_threshold=0.80 is a single
        # dataset-agnostic value and has no recorded calibration rationale.
        # As of 2026-04-22, both datasets clear this threshold comfortably
        # (Amazon AUROC=0.9744, YelpChi AUROC=0.9867; see
        # status_package/general.md §"Task 2 运行时验收" for the sha256-pinned
        # baseline hashes of canonical .mat + checkpoint that produced these).
        # scripts/verify_teacher_metric_consistency.py must be re-run if any of
        # those hashes changes. Do NOT edit the two lines below without (a) a
        # verification-run log, and (b) a per-dataset rationale committed to
        # status_package/general.md.
        metric_name=MetricName.AUROC,
        metric_threshold=0.80,
        population_name=PopulationName.VALIDATION,
        validation_ground_truth_label=val_labels,
        validation_teacher_prob=val_probs,
        code_git_sha=code_git_sha,
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
            code_git_sha=code_git_sha,
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

    # Resolve HEAD SHA once, fail-closed, before any artifact is written.
    code_git_sha = _git_head_sha_or_fail(BRIDGE_FILES)
    print(f"[provenance] code_git_sha = {code_git_sha}")

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        generate_for_dataset(
            ds, output_dir, device=args.device, code_git_sha=code_git_sha
        )


if __name__ == "__main__":
    main()
