"""Head-only inference smoke test on validation subset (first 16 rows).

Uses ONLY validation data. Does NOT access final_test.
Loads backbone + PEFT adapter + cls_head, builds student-visible Evidence Cards,
runs head-only inference via hidden-state extraction and sigmoid.
Follows the canonical path from eval.head_scoring.score_head.
"""

from __future__ import annotations

import sys
import os

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from evidence.evidence_schema import build_student_evidence_card
from evidence.prompt_builder import PromptMode, ThinkingMode, build_prompt
from llm.hidden_state_pooling import pool_last_valid_token
from priorf_teacher.schema import (
    DatasetName,
    GraphRegime,
    NeighborSummary,
    PopulationName,
    RelationProfile,
    TeacherExportRecord,
)


class ClsHeadModule(nn.Module):
    """Minimal classification head matching the saved state dict keys."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2560, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def main() -> None:
    # ---------------------------------------------------------------- paths
    backbone_path = "/data1/mq/models/Qwen3-4B-Instruct-2507"
    adapter_path = (
        "outputs/verification/stage2/"
        "amazon_stage9_7_stage2_diagnostic_full/best_checkpoint/peft_adapter"
    )
    cls_head_path = (
        "outputs/verification/stage2/"
        "amazon_stage9_7_stage2_diagnostic_full/best_checkpoint/cls_head.pt"
    )
    teacher_export_path = (
        "outputs/gated/teacher_exports/amazon/validation/teacher_export.parquet"
    )
    device = "cuda:0"

    # --------------------------------------------------------- load tokenizer
    print("[1/6] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(backbone_path)

    # --------------------------------------------------- load backbone model
    print("[2/6] Loading backbone model ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        backbone_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    base_model.eval()

    # ------------------------------------------------ load PEFT adapter
    print("[3/6] Loading PEFT adapter ...")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        autocast_adapter_dtype=False,
    )
    model.eval()

    # ------------------------------------------------ load cls_head
    print("[4/6] Loading cls_head ...")
    cls_head_state = torch.load(cls_head_path, map_location="cpu", weights_only=True)
    cls_head = ClsHeadModule()
    cls_head.load_state_dict(cls_head_state)
    cls_head_dtype = torch.bfloat16
    cls_head.to(device=device, dtype=cls_head_dtype)
    cls_head.eval()

    print(f"    cls_head loaded: weight shape={cls_head.linear.weight.shape}, "
          f"dtype={cls_head.linear.weight.dtype}")

    # ------------------------------------------------ load teacher export
    print("[5/6] Loading validation teacher export ...")
    df = pd.read_parquet(teacher_export_path)
    print(f"    Full validation set: {len(df)} rows")
    # Take a stratified subset: first 8 negatives + first 8 positives
    # so that both classes are present for AUROC/AUPRC computation.
    neg = df[df["ground_truth_label"] == 0].head(8)
    pos = df[df["ground_truth_label"] == 1].head(8)
    subset = pd.concat([neg, pos]).sort_index()
    print(f"    Using stratified subset: {len(neg)} negative + {len(pos)} positive = {len(subset)} rows")

    # minimal data manifest-like object for build_student_evidence_card
    class _Manifest:
        dataset_name = "amazon"
        graph_regime = "transductive_standard"
        num_nodes = 11944  # Amazon total nodes

    data_manifest = _Manifest()

    # ------------------------------------------- inference loop
    print("[6/6] Running head-only inference on 16 samples ...")
    probs = []
    labels = []

    with torch.inference_mode():
        for idx, row in subset.iterrows():
            # Build TeacherExportRecord from parquet row
            rel_dict = row["relation_profile"]
            nbr_dict = row["neighbor_summary"]
            record = TeacherExportRecord(
                dataset_name=DatasetName(row["dataset_name"]),
                teacher_model_name=row["teacher_model_name"],
                teacher_checkpoint=row["teacher_checkpoint"],
                population_name=PopulationName(row["population_name"]),
                node_id=int(row["node_id"]),
                ground_truth_label=int(row["ground_truth_label"]),
                teacher_prob=float(row["teacher_prob"]),
                teacher_logit=float(row["teacher_logit"]),
                hsd=float(row["hsd"]),
                hsd_quantile=float(row["hsd_quantile"]),
                asda_switch=bool(row["asda_switch"]),
                mlp_logit=float(row["mlp_logit"]),
                gnn_logit=float(row["gnn_logit"]),
                branch_gap=float(row["branch_gap"]),
                relation_profile=RelationProfile(**rel_dict),
                neighbor_summary=NeighborSummary(**nbr_dict),
                high_hsd_flag=bool(row["high_hsd_flag"]),
                graph_regime=GraphRegime(row["graph_regime"]),
            )

            # Build student-visible Evidence Card
            card = build_student_evidence_card(
                teacher_record=record,
                data_manifest=data_manifest,
            )

            # Build prompt (EVAL_HEAD mode)
            bundle = build_prompt(
                evidence_card=card,
                mode=PromptMode.EVAL_HEAD,
                thinking_mode=ThinkingMode.NON_THINKING,
            )

            # Convert ChatMessages to list-of-dicts for apply_chat_template
            messages = [
                {"role": m.role, "content": m.content}
                for m in bundle.messages
            ]

            # Tokenize with chat template (canonical path from head_scoring)
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=False,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            last_hidden = outputs.hidden_states[-1]
            pooled = pool_last_valid_token(last_hidden, attention_mask)

            logit = cls_head(pooled).to(torch.float32)
            prob = torch.sigmoid(logit).squeeze().item()

            gt = int(row["ground_truth_label"])
            probs.append(prob)
            labels.append(gt)

            print(f"    sample {idx:>4d} | node_id={record.node_id:>5d} | "
                  f"gt={gt} | prob={prob:.6f}")

    # ------------------------------------------- aggregate report
    print("\n" + "=" * 60)
    print("HEAD-ONLY SMOKE TEST RESULTS (validation subset, n=16)")
    print("=" * 60)

    probs_arr = np.array(probs, dtype=np.float64)
    labels_arr = np.array(labels, dtype=np.int64)

    print(f"\nPer-sample prob stats:")
    print(f"  mean  = {probs_arr.mean():.6f}")
    print(f"  std   = {probs_arr.std():.6f}")
    print(f"  min   = {probs_arr.min():.6f}")
    print(f"  max   = {probs_arr.max():.6f}")

    print(f"\nLabel distribution:")
    print(f"  negative (0) = {(labels_arr == 0).sum()}")
    print(f"  positive (1) = {(labels_arr == 1).sum()}")

    # Near-constant check
    if probs_arr.std() < 0.001:
        print("\n  *** NEAR-CONSTANT WARNING: prob_std < 0.001 ***")
    else:
        print(f"\n  Near-constant check: PASS (std={probs_arr.std():.6f} >= 0.001)")

    # AUROC / AUPRC if both classes present
    if len(np.unique(labels_arr)) >= 2:
        auroc = roc_auc_score(labels_arr, probs_arr)
        auprc = average_precision_score(labels_arr, probs_arr)
        print(f"\n  AUROC = {auroc:.6f}")
        print(f"  AUPRC = {auprc:.6f}")
    else:
        print("\n  AUROC / AUPRC: SKIPPED (single class in subset)")

    print("\n" + "=" * 60)
    print("SMOKE TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
