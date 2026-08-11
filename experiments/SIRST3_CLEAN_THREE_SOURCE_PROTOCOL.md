# EviSIRST SIRST3 clean joint-training protocol

## Frozen question

Train one clean EviSIRST model on the pooled SIRST3 train split, then reuse the
same fixed endpoint on the held-out NUAA-SIRST, NUDT-SIRST, and IRSTD-1K test
splits. This is multi-source joint training followed by source-specific testing;
it is not zero-shot cross-dataset evaluation.

## Model and training

- Model: `EviSIRST` (`TPD8-MPRS-DCH + NER4 Tail-Aware + QFG2-CROA`).
- Graph: 564 state keys, 10,870,130 parameters, no TSS module/state/loss.
- Initialization: scratch, seed 42.
- Train data: only `SIRST3/img_idx/train_SIRST3.txt` (1676 images).
- Source order: NUAA 213, NUDT 663, IRSTD 800; natural-frequency shuffle.
- Normalization: mean `101.06385040283203`, std `34.619606018066406`.
- Epochs: 1000; batch: 16; patch: 256; workers: 0; FP32; TF32 off.
- Optimizer: Adam, base LR `1e-3`, no weight decay.
- Schedule: epoch 1–10 linear `1e-4 → 1e-3`; cosine to `1e-5` at epoch 1000.
- Loss: sum of six equal-weight mean BCELoss terms.
- Crop/augmentation: historical source-namespaced stateless plan.
- DataLoader shuffle: historical length-prefixed `stable_uint63` seed.

Training must not open any test index/image/mask and must not compute test
metrics. `last_training_state.pth.tar` is recovery state only.

## Frozen endpoint and evaluation

- Candidate: epoch 1000 `EviSIRST.pth.tar` only.
- Checkpoint selection: fixed endpoint; no best checkpoint and no positive-gain
  threshold.
- Freeze checkpoint SHA-256 before any target test is evaluated.
- Reuse exactly that checkpoint on:
  - NUAA-SIRST test: 214 images;
  - NUDT-SIRST test: 664 images;
  - IRSTD-1K test: 201 images.
- All three tests retain SIRST3 normalization; target-specific normalization is
  forbidden in the main protocol.
- Prediction: probability strictly `> 0.5`; no threshold sweep/post-hoc tuning.
- Metrics: global foreground mIoU, image-normalized IoU, pixel precision/recall/F1,
  8-connected one-to-one Pd/tiny-Pd with centroid distance strictly `< 3`, and
  unmatched-component-pixel Fa.

## Frozen data identity

- SIRST3 train index file SHA-256:
  `75c32b896b95e29b89edc1f5231f619f275c2b54da0264934e6e0df13d7e7d9a`
- SIRST3 train ordered-ID SHA-256:
  `66c5a6f43b665e36556a97391c1676e3720b3ae0e72186b4894a9eadb6456355`
- Preflight train image/mask tree SHA-256:
  `d0dda06a2c8e1f1617c3d2637aa4cde4342f643d55c373bcdccae3b695dd41fe`
- The SIRST3 train list is the strict ordered concatenation of the three source
  train lists, with zero duplicate IDs.

## Outputs

- Recovery artifacts: `runs/sirst3_clean_v1/SIRST3/`.
- Fixed checkpoint: `results/SIRST3/EviSIRST.pth.tar`.
- Per-source JSON: `results/SIRST3/evaluations/<dataset>.json`.
- Aggregate JSON/CSV/Markdown: `results/SIRST3/cross_dataset_results.*`.

The launcher is `run_sirst3_experiment.py`. It validates the epoch, graph,
training identity, index hash, absence of TSS, and test-selection flags before
allowing the three evaluations.

## Interpretation boundary

Historical SIRST3 Final results used a 568-key training graph with a nonzero TSS
auxiliary loss and selected epoch 600 on SIRST3 test. They are provenance-only
references, not direct results for this clean fixed-endpoint experiment.
