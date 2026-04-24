# Self-review: QA + hyperparameter-tuned Stage 2 training

Covers the work after the initial end-to-end pipeline delivery: QA pass
over every new script, root-cause analysis of the noisy loss observed in
run_v1, targeted trainer upgrades, and a tmux-backed retraining run
(`run_v2`) on both Amazon and YelpChi.

## 1. QA pass

Tooling:
- `pyflakes`, `ruff check ... --select=F,B,UP,SIM,ARG` over the five new
  scripts.
- Manual contract cross-check against `eval.head_scoring.score_head`,
  `train.train_stage2_canonical.run_canonical_train_step`,
  `eval.eval_head_only.run_formal_head_only_eval`,
  `evidence.evidence_schema.build_evidence_card`.

Findings + fixes:

- `scripts/run_stage2_train.py` imported `eval.head_scoring.score_head`
  which was never called.  Removed.
- `scripts/run_head_only_smoke.py` had two separate
  `from priorf_teacher.schema import ...` statements separated by an
  unrelated import.  Collapsed to a single grouped import.
- E501 (line length > 88) was the only other rule tripped; kept
  everywhere in the driver scripts since enforcing 88 for argparse
  help strings would degrade readability.  Not a lint category I
  promote here.
- Contract check: every call site passes exactly the keyword argument
  names expected by the public canonical function; no positional
  reordering.  `CheckpointProvenance.content_hash` is 64-hex; the
  in-loop monitor passes a 64-character sentinel (`"m" * 64`) and
  never promotes that provenance - the saved final checkpoint and the
  formal eval use the real sha256 of `cls_head.pt`.

216 tests still pass after the QA edits.

## 2. Root cause of `run_v1` poor convergence

`run_v1` trained for 40 steps at bs=2 with a flat lr=1e-4 over a
random-with-replacement sampler on a 1024-sample cap, LoRA on
`{q_proj, v_proj}` only.  Symptoms:

- `L_cls` collapsed to ~0.05 within the first 4 steps (over-fitting a
  small random subset).
- `L_distill` oscillated wildly step-to-step (0.36 vs 1.14 adjacent
  steps) because there was no warmup and LR was already at peak.
- Post-training validation `prob_std = 0.0040` (Amazon) /
  `0.0130` (YelpChi); predictions collapsed into a tiny band around
  the prior, so even though the relative ordering was correct, the
  head was effectively non-discriminating.

These are all consistent with three issues:

1. no LR schedule (no warmup, no decay),
2. tiny trainable rank (LoRA targets only `{q, v}`),
3. random-with-replacement sampler that duplicated positives / negatives
   unevenly and visited only a small fraction of the cap.

## 3. Trainer upgrades (committed in `797a641`)

Exclusively in `scripts/run_stage2_train.py`.  No canonical module was
touched.

- `--warmup-ratio` (default 0.1) + `--lr-scheduler {cosine,linear,constant}`.
  Uses the transformers schedulers; step 0 applies near-zero LR during
  warmup.
- `--lora-targets` default expanded from `{q_proj, v_proj}` to the
  attention + MLP set
  `{q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj}`,
  driving trainable parameters from 2.9 M to 33 M (out of 4.03 B).
  Added `--lora-dropout` (default 0.05) and `--weight-decay`.
- Epoch-based iteration: each epoch produces a fresh random permutation
  of `train_samples`; batches pop from the permutation until exhausted,
  then reshuffle.  Every sample is now seen at least once per epoch
  before any repeat.
- `--num-epochs` computes `total_steps = ceil(epochs * n_train / batch_size)`;
  `--max-steps` still wins if both are provided; raises if neither set.
- Periodic in-loop validation: `--validation-every-n-steps N` calls
  `run_validation_with_unified_scorer` on a pre-loaded stratified
  subset every N steps.  Each monitor row is written to
  `train_log.jsonl` alongside the loss rows, and the interim provenance
  uses a sentinel `content_hash` so it cannot be confused for the final
  checkpoint bundle.
- Per-step LR logged to `train_log.jsonl`.

## 4. `run_v2` training (tmux-backed)

Two tmux sessions on GPU 0 (Amazon) and GPU 1 (YelpChi), same config:

```
--max-steps 500 --batch-size 2 --max-train-samples 4096
--learning-rate 1e-4 --warmup-ratio 0.1 --lr-scheduler cosine
--lora-r 16 --lora-alpha 32 --lora-dropout 0.05
--lambda-cls 1.0 --lambda-distill 0.3
--validation-subset 128 --validation-every-n-steps 50
```

Rationale:

- 500 steps at bs=2 over a 4096-sample cap leaves ~half an epoch per
  step, which is enough for the training loop to touch a wide variety
  of positives and negatives without spending hours.
- `lambda_distill = 0.3` (was 0.5) damps the distill-loss swings
  that were visible in `run_v1`.
- `lora_r = 16` gives more capacity on top of the broader targets.
- Warmup = 10 % of total_steps (50 steps) keeps the LR below peak
  while the randomly-initialized `cls_head` starts learning.

Amazon validation-AUROC trajectory (monitored on 128 stratified rows):

```
step  50: 0.5584
step 100: 0.6078
step 150: 0.6634
step 200: 0.6625
step 250: 0.7876
step 300: 0.9482
step 350: 0.9888
step 400: 0.9953
step 450: 0.9944
step 500: 0.9925
post-training (full 128): 0.9337
```

YelpChi validation-AUROC trajectory:

```
step  50: 0.5995
step 100: 0.7064
step 150: 0.8141
step 200: 0.8402
step 250: 0.8757
step 300: 0.8324
step 350: 0.8503
step 400: 0.9558
step 450: 0.9681
step 500: 0.9701
post-training (full 128): 0.9797
```

Loss trajectories are smooth and strictly decreasing on the aggregate
trend (individual step spikes are expected with bs=2): `L_gen` goes
from ~2.75 to ~0.31 on both datasets, `L_cls` settles around 0.01 - 0.1
with occasional spikes when the batch is all-negative, `L_distill`
settles around 0.05 - 0.7 (half the previous mean because of the
lowered weight).  `prob_std` grows monotonically from ~0.001 to
~0.06 - 0.10 (+25x to +80x vs `run_v1`), recovering real discriminative
spread.

## 5. Final formal head-only eval

| | Amazon run_v1 | Amazon **run_v2** | YelpChi run_v1 | YelpChi **run_v2** |
|---|---|---|---|---|
| val-frozen threshold | 0.0647 | **0.2082** | 0.1394 | **0.1112** |
| final_test AUROC | 0.5093 | **0.9857** | 0.6287 | **0.9429** |
| final_test AUPRC | 0.0882 | **0.9051** | 0.1889 | **0.7233** |
| F1 @ val_thr | 0.1053 | **0.8696** | 0.1818 | **0.7368** |
| precision @ val_thr | 0.0690 | **0.8824** | 0.1750 | **0.7179** |
| recall @ val_thr | 0.2222 | **0.8571** | 0.1892 | **0.7568** |
| specificity @ val_thr | -- | **0.9916** | -- | **0.9498** |
| Brier | 0.0654 | **0.0331** | 0.1229 | **0.1097** |
| ECE | -- | 0.0783 | -- | 0.1035 |

Amazon AUROC +48 points, YelpChi AUROC +31 points absolute, both
formal reports pass every `FormalHeadOnlyReport` model_validator
(population_metadata / tuning-rows contract, scorer provenance
bundle identity, calibration population alignment, validation
threshold source).

## 6. Known limitations

- `validation_threshold.source_population_name = validation`, but the
  validation population here still contains the teacher that produced
  these Evidence Cards.  This is consistent with the declared
  `DataManifest.populations[...].contains_tuning_rows = True`, so the
  formal runner accepts it, but any claim of "generalization beyond
  teacher distribution" requires a held-out external test population.
- ECE of 0.08 (Amazon) / 0.10 (YelpChi) and `max_calibration_gap` of
  0.56 / 0.79 indicate the head is well-ordered but under-calibrated.
  If downstream fusion / gating needs well-calibrated probabilities,
  follow-up temperature scaling on validation is cheap and
  principled.
- Training was still capped at 4096 samples with 500 optimizer steps.
  A larger budget (a full epoch over the TRAIN parquet at bs=2 is
  ~4-16k steps depending on dataset) combined with best-checkpoint
  selection on the validation scorer should close the remaining gap
  between `run_v2` and a teacher-parity student.
