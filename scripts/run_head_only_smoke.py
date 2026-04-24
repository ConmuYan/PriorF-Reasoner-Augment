"""Diagnostic-only head-only smoke runner on real Qwen3-4B + legacy teacher exports.

What this script IS:

* a **diagnostic** end-to-end plumbing smoke that exercises the canonical
  Task 3 / Task 4 / Task 5 stack on real Qwen3-4B weights and a subset of
  real teacher-derived Evidence Cards,
* a path that routes every output under ``outputs/diagnostic/head_only_smoke/``,
* a stress test for ``EvidenceCard`` → ``build_prompt`` → Qwen chat template
  → forward with ``output_hidden_states=True`` → ``pool_last_valid_token`` →
  ``cls_head`` → ``torch.sigmoid`` → ``ScorerReport``.

What this script IS NOT:

* formal evaluation.  The cls_head is randomly initialized; no Stage 2
  training has happened; the reported AUROC / AUPRC / Brier are meaningless
  for quality assessment and explicitly not for gate promotion.
* a replacement for ``scripts/generate_teacher_exports.py`` or the canonical
  ``TeacherExportRecord`` pipeline.  Legacy parquet columns are adapted
  one-to-one into EvidenceCard fields using conservative, clearly-synthetic
  defaults where legacy schema has no direct match (notably
  ``NeighborSummary`` label counts).  The resulting EvidenceCards are
  labelled ``population_name=diagnostic_holdout`` so they cannot be confused
  with formal populations.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.head_scoring import (  # noqa: E402
    CheckpointProvenance,
    HeadScoringInputs,
    HeadScoringSample,
    ScorerReport,
    score_head,
)
from evidence.evidence_schema import (  # noqa: E402
    CANONICAL_SCHEMA_HINT_ORDER,
    DiscrepancySummary,
    EvidenceCard,
    TaskInstruction,
    TeacherSummary,
)
from priorf_teacher.schema import NeighborSummary, RelationProfile  # noqa: E402
from evidence.prompt_builder import ThinkingMode  # noqa: E402
from priorf_teacher.schema import DatasetName, GraphRegime, PopulationName  # noqa: E402


_HSD_QUANTILE_MAP: Final[dict[str, float]] = {
    "normal": 0.5,
    "top_10_percent": 0.95,
    "top_5_percent": 0.975,
    "top_1_percent": 0.995,
}

# Legacy parquet column naming differs per dataset: Amazon uses U-based
# relation suffixes (UPU / USU / UVU), YelpChi uses R-based (RUR / RTR / RSR).
_RELATION_SUFFIXES: Final[dict[DatasetName, tuple[str, str, str]]] = {
    DatasetName.AMAZON: ("UPU", "USU", "UVU"),
    DatasetName.YELPCHI: ("RUR", "RTR", "RSR"),
}

_NAMESPACE_ROOT: Final[str] = "outputs/diagnostic/head_only_smoke"
_DIAGNOSTIC_POPULATION: Final[PopulationName] = PopulationName.DIAGNOSTIC_HOLDOUT


def _row_to_evidence_card(row: pd.Series, *, dataset: DatasetName) -> EvidenceCard:
    """Adapt one legacy parquet row to a schema-valid EvidenceCard.

    This is diagnostic-only.  The adapter synthesises NeighborSummary
    label counts from ``suspicious_neighbor_ratio`` because the legacy
    schema does not carry explicit labeled / unlabeled counts.  All other
    EvidenceCard fields are read directly from the legacy columns.
    """
    teacher_summary = TeacherSummary(
        teacher_prob=float(row["teacher_prob"]),
        teacher_logit=float(row["teacher_logit"]),
        hsd=float(row["hsd"]),
        hsd_quantile=_HSD_QUANTILE_MAP[str(row["hsd_quantile"])],
        asda_switch=bool(row["asda_switch"] >= 0.5),
        mlp_logit=float(row["mlp_logit"]),
        gnn_logit=float(row["gnn_logit"]),
        branch_gap=float(row["branch_gap"]),
        high_hsd_flag=bool(row["high_hsd_flag"]),
    )

    branch_gap_abs = abs(float(row["branch_gap"]))
    if branch_gap_abs < 0.5:
        severity = "low"
    elif branch_gap_abs < 1.5:
        severity = "medium"
    else:
        severity = "high"
    mlp_logit = float(row["mlp_logit"])
    gnn_logit = float(row["gnn_logit"])
    teacher_logit = float(row["teacher_logit"])
    teacher_mlp_agreement = (teacher_logit >= 0.0) == (mlp_logit >= 0.0)
    teacher_gnn_agreement = (teacher_logit >= 0.0) == (gnn_logit >= 0.0)
    gap = abs(mlp_logit - gnn_logit)
    if gap < 0.25:
        route_hint = "balanced"
    elif mlp_logit > gnn_logit:
        route_hint = "mlp_dominant"
    else:
        route_hint = "gnn_dominant"
    discrepancy_summary = DiscrepancySummary(
        branch_gap_abs=branch_gap_abs,
        teacher_mlp_agreement=teacher_mlp_agreement,
        teacher_gnn_agreement=teacher_gnn_agreement,
        discrepancy_severity=severity,
        route_hint=route_hint,
    )

    suffixes = _RELATION_SUFFIXES[dataset]
    degrees = [int(row[f"degree_{s}"]) for s in suffixes]
    discs = [float(row[f"disc_{s}"]) for s in suffixes]
    relation_profile = RelationProfile(
        total_relations=3,
        active_relations=sum(1 for d in degrees if d > 0),
        max_relation_neighbor_count=max(degrees),
        mean_relation_neighbor_count=float(np.mean(degrees)),
        max_relation_discrepancy=max(discs),
        mean_relation_discrepancy=float(np.mean(discs)),
    )

    total_neighbors = sum(degrees)
    suspicious_ratio = float(row["suspicious_neighbor_ratio"])
    positive_neighbors = int(round(suspicious_ratio * total_neighbors))
    negative_neighbors = total_neighbors - positive_neighbors
    labeled_neighbors = positive_neighbors + negative_neighbors
    unlabeled_neighbors = total_neighbors - labeled_neighbors
    neighbor_summary = NeighborSummary(
        total_neighbors=total_neighbors,
        labeled_neighbors=labeled_neighbors,
        positive_neighbors=positive_neighbors,
        negative_neighbors=negative_neighbors,
        unlabeled_neighbors=unlabeled_neighbors,
    )

    return EvidenceCard(
        dataset_name=dataset,
        population_name=_DIAGNOSTIC_POPULATION,
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
        node_id=int(row["node_id"]),
        teacher_summary=teacher_summary,
        discrepancy_summary=discrepancy_summary,
        relation_profile=relation_profile,
        neighbor_summary=neighbor_summary,
        task_instruction=TaskInstruction(
            text=(
                "Use the structural Evidence Card under the declared graph "
                "regime to produce the strict JSON output."
            ),
            schema_hint_order=CANONICAL_SCHEMA_HINT_ORDER,
        ),
    )


def _stratified_subset(df: pd.DataFrame, *, subset_size: int, seed: int) -> pd.DataFrame:
    """Return a stratified (by label) subset of the legacy parquet dataframe."""
    if subset_size >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    pos_df = df[df["label"] == 1]
    neg_df = df[df["label"] == 0]
    pos_rate = len(pos_df) / len(df)
    n_pos = max(1, int(round(subset_size * pos_rate)))
    n_pos = min(n_pos, len(pos_df))
    n_neg = subset_size - n_pos
    n_neg = min(n_neg, len(neg_df))
    pos_sample = pos_df.sample(n=n_pos, random_state=seed)
    neg_sample = neg_df.sample(n=n_neg, random_state=seed)
    combined = pd.concat([pos_sample, neg_sample], ignore_index=True)
    return combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)


class _RandomLinearClsHead(torch.nn.Module):
    """Diagnostic random-init linear head used to exercise the scorer path.

    Real formal use requires a Stage-2-trained head; this class exists so
    the Qwen3-4B forward + ``pool_last_valid_token`` + ``torch.sigmoid``
    chain can be validated before training.  Outputs are uninformative by
    construction.
    """

    def __init__(self, hidden_size: int, *, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.weight = torch.nn.Parameter(
            torch.empty(hidden_size, 1).normal_(generator=generator) * 0.01
        )
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, hidden_prompt_only: torch.Tensor) -> torch.Tensor:
        # score_head requires a 1-D logit tensor of length B; pooled shape is
        # [B, H] so we return [B] via a squeeze on the last dim.
        logits_2d = hidden_prompt_only.to(self.weight.dtype) @ self.weight + self.bias
        return logits_2d.squeeze(-1)

    __call__ = forward


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--legacy-parquet",
        required=True,
        type=Path,
        help="Path to legacy teacher export parquet (assets/teacher_exports/*.parquet).",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["amazon", "yelpchi"],
        help="Dataset name (controls DatasetName enum).",
    )
    parser.add_argument(
        "--qwen-path",
        required=True,
        type=Path,
        help="Path to a local Qwen3-style checkpoint (e.g. /data1/mq/models/Qwen3-4B-Instruct-2507).",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=16,
        help="Number of legacy rows (stratified by label) to feed through score_head.",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="GPU index to use.  -1 selects CPU.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Root output directory.  Reports land under <root>/diagnostic/head_only_smoke.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    args = _parse_args(argv)

    dataset_enum = DatasetName(args.dataset)

    print(f"[1/6] Loading legacy parquet: {args.legacy_parquet}", flush=True)
    df = pd.read_parquet(args.legacy_parquet)
    subset_df = _stratified_subset(df, subset_size=args.subset_size, seed=args.seed)
    print(
        f"       kept {len(subset_df)} rows "
        f"(positives={(subset_df['label'] == 1).sum()}, "
        f"negatives={(subset_df['label'] == 0).sum()})",
        flush=True,
    )

    print("[2/6] Building EvidenceCards from legacy rows...", flush=True)
    samples: list[HeadScoringSample] = []
    for _, row in subset_df.iterrows():
        card = _row_to_evidence_card(row, dataset=dataset_enum)
        samples.append(
            HeadScoringSample(
                evidence_card=card,
                ground_truth_label=int(row["label"]),
                node_id=int(row["node_id"]),
            )
        )

    device_str = "cpu" if args.gpu_index < 0 else f"cuda:{args.gpu_index}"
    print(f"[3/6] Loading Qwen3 checkpoint on {device_str}: {args.qwen_path}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

    tokenizer = AutoTokenizer.from_pretrained(str(args.qwen_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(args.qwen_path),
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.to(device_str)
    model.eval()
    hidden_size = int(model.config.hidden_size)

    print(f"[4/6] Building random-init cls_head with hidden_size={hidden_size}", flush=True)
    cls_head = _RandomLinearClsHead(hidden_size=hidden_size, seed=args.seed)
    cls_head.to(device_str)
    cls_head.eval()

    checkpoint_provenance = CheckpointProvenance(
        path=f"diagnostic://head_only_smoke/random_linear_head/seed{args.seed}",
        step=0,
        content_hash="d" * 64,  # diagnostic sentinel; not a real sha256
    )
    inputs = HeadScoringInputs(
        samples=tuple(samples),
        dataset_name=dataset_enum,
        population_name=_DIAGNOSTIC_POPULATION,
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
        checkpoint_provenance=checkpoint_provenance,
    )

    print(f"[5/6] Running score_head on {len(samples)} samples...", flush=True)
    report: ScorerReport = score_head(
        inputs=inputs,
        model=model,
        cls_head=cls_head,
        tokenizer=tokenizer,
        thinking_mode=ThinkingMode.NON_THINKING,
        accelerator=None,
    )

    print("[6/6] Writing ScorerReport JSON to diagnostic namespace...", flush=True)
    output_dir = args.output_root / "diagnostic" / "head_only_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"scorer_report_{dataset_enum.value}_{args.subset_size}.json"
    report_payload: dict[str, Any] = json.loads(report.model_dump_json())
    report_payload["_diagnostic_provenance"] = {
        "legacy_parquet": str(args.legacy_parquet),
        "legacy_parquet_note": "legacy schema; not TeacherExportRecord",
        "cls_head": "random_linear; pre-training; numbers uninformative",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path.write_text(json.dumps(report_payload, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"SMOKE OK: wrote {report_path}")
    print(
        f"  n_total={report.n_total}  n_positive={report.n_positive}  "
        f"n_negative={report.n_negative}  is_single_class={report.is_single_class_population}"
    )
    print(
        f"  auroc={report.auroc}  auprc={report.auprc}  brier={report.brier_score}"
    )
    print(
        f"  prob_mean={report.prob_mean:.4f}  prob_std={report.prob_std:.4f}  "
        f"min={report.prob_min:.4f}  max={report.prob_max:.4f}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
