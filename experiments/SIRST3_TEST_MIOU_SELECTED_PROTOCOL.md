# EviSIRST SIRST3 test-mIoU-selected protocol

## Purpose

This protocol reproduces the operational checkpoint-selection rule used by the
SCTransNet repository for its pooled SIRST3 experiment. It is separate from the
unbiased fixed-epoch protocol in `SIRST3_CLEAN_THREE_SOURCE_PROTOCOL.md`.

The selected result is **optimistic**: the same pooled test set is used both to
choose the epoch and to report the final source-specific metrics. It must be
labelled `test_selected=true`; it is suitable for reproducing the repository's
reported convention, not for estimating unseen-test generalization.

## Frozen training recipe

- Model: clean 564-key EviSIRST inference/training graph, with no TSS module.
- Training set: `train_SIRST3`, 1,676 samples in the fixed order
  NUAA-SIRST (213), NUDT-SIRST (663), IRSTD-1K (800).
- Seed: 42; batch size: 16; crop: 256; workers: 0; FP32; no AMP.
- Optimizer: Adam, base LR `1e-3`.
- Schedule: 10-epoch linear warmup followed by cosine decay to `1e-5`.
- Loss: unweighted sum of six mean BCE deep-supervision terms.
- Normalization for training and every selection/test image: SIRST3
  (`mean=101.06385040283203`, `std=34.619606018066406`).
- Frozen train image/mask tree SHA-256:
  `0593df78057dbc89679ed7a1d1c14a837cd537b96371fbb4c577273dacf19f44`.
- Frozen pooled-selection image/mask tree SHA-256:
  `aac23b1df82514057f0b2bc573499f80defc257ee436929277f49ff3b7619d7b`.
- Frozen training-source tree SHA-256:
  `a48630f86ec86950aa150db4d1e4a86b9c125c5029bf3484659e1523f2bc0a53`.

## Checkpoint selection

After each completed epoch from 500 through 1000, inclusive:

1. Evaluate the same in-memory model on the ordered pooled SIRST3 test set,
   1,079 images = NUAA-SIRST 214 + NUDT-SIRST 664 + IRSTD-1K 201.
2. Threshold the probability map with strict `> 0.5`.
3. Accumulate foreground intersection and union over all 1,079 images using the
   exact histogram semantics of the SCTransNet repository's
   `batch_intersection_union` implementation.
4. Compute one global foreground mIoU, not the arithmetic mean of three source
   mIoUs.
5. Replace the selected candidate only when the raw floating-point mIoU is
   strictly greater. Exact ties retain the earlier epoch.

The first comparison uses an incumbent score of zero, matching the repository.
No threshold search or additional performance margin is used.

## Artifacts

The training directory is
`runs/sirst3_test_miou_selected_v1/SIRST3/` and contains:

- `best_selection_state.pth.tar`: selected checkpoint plus optimizer/RNG state
  for crash-safe recovery;
- `last_training_state.pth.tar`: latest completed epoch recovery state;
- `EviSIRST_SIRST3_test_mIoU_selected.pth.tar`: slim selected checkpoint;
- `EviSIRST_epoch1000.pth.tar`: slim fixed endpoint from the same run;
- `summary.json`: complete selection history and provenance.

Final three-source evaluation artifacts are written under
`results/SIRST3/test_mIoU_selected/`. The existing fixed-epoch result under
`results/SIRST3/` is not overwritten.

## Entry point

```bash
CUDA_VISIBLE_DEVICES=2 \
  /home/ly/BasicIRSTD/infrarenet/bin/python \
  run_sirst3_test_selected_experiment.py \
  --dataset-root /home/ly/SCTransNet_main/datasets \
  --device cuda:0
```

The runner resumes only from artifacts with matching source, runtime, data-tree,
training, and selection identities. The selected checkpoint is then evaluated
separately on NUAA-SIRST, NUDT-SIRST, and IRSTD-1K with the common public
evaluator.
