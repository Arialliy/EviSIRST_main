# IRSTD-1K HF Decoder V1 frozen protocol

Status: preregistered, validation-only, single-variable Stage-A pilot.

## Question and hypotheses

The paired clean control is the validation-selected EviSIRST R1 run with
`run_seed=1446202191`.  Its comparison checkpoint is recomputed offline from
the complete saved validation history with the new zero-margin, complete key;
it is not copied from the historical `0.001`-window summary.  Any interim
winner observed before that control reaches its frozen terminal epoch is not
the final comparator and must not be hard-coded into this variant's run
identity.  HF Decoder V1 asks whether one small, identity-starting
high-frequency residual on the final 32-channel decoder feature improves
IRSTD-1K validation overlap while leaving every other training choice fixed.

- **H0:** the variant's selected validation mIoU is less than or equal to the
  zero-margin reselected paired clean control's validation mIoU.
- **H1:** the variant's selected validation mIoU is strictly greater than the
  zero-margin reselected paired clean control's validation mIoU.

There is no effect-size threshold and no `0.001` candidate window.  A positive
raw difference is evidence for H1; zero or a negative difference supports the
STOP decision.

## Immutable comparison contract

- Dataset and target: `IRSTD-1K`, binary masks.
- Split: the repository's frozen V2 manifest, exactly 640 train and 160
  validation samples.  Sample-level fallback must be acknowledged explicitly.
- Architecture seed: 42.  Runtime seed: 1446202191.
- Initialization: the complete base model and the HF component are initialized
  from scratch; no checkpoint is loaded before epoch 1.
- Optimization: all base-model and HF-component parameters are trainable in one
  Adam group, using the clean-control learning-rate recipe.
- Training: 1000 epochs, batch size 16, clean R1 crop/augmentation, six equally
  weighted BCE probability heads, no complete-target crop, modified loss, or
  center head.
- Validation: every epoch, fixed `out` head, fixed threshold and matching
  metrics inherited from validation-selected R1.
- Selection: validation only, zero margin.  `best_mIoU` uses the complete
  strict deterministic key supplied by `evisirst_zero_margin_selection`; a
  separate `best_Pd` role is retained and reported.  No tolerance window is
  permitted.
- Test: unavailable to this runner.  No test index, test dataset, test loader,
  test metric, or public-test bridge may be imported, constructed, or opened.
- Output: isolated under `runs/irstd_performance/hf_decoder_v1/`.

The only experimental change relative to the paired clean control is the
identity-starting HF decoder residual defined in
`experiments/irstd_hf_decoder_v1.py`.

## Decision rule

- **GO-1:** selected validation mIoU is strictly greater than the zero-margin
  reselected paired clean control (`variant - clean > 0`).
- **GO-2:** selected validation mIoU is also compared with the complete-target
  run after its complete saved validation history is independently reselected
  by this same zero-margin complete key (`variant - complete-target > 0`).  The
  old complete-target summary's `0.001`-window winner is not a valid comparator.
- Report the complete metric key, including nIoU, F1, Pd, Fa, tiny-Pd, loss,
  and epoch, so any trade-off remains visible.
- **STOP:** selected validation mIoU is equal to or below the paired clean
  control.  Do not stack another component onto this failed branch.

The single-seed result is a screening result, not a publication claim.  A GO
permits a separately preregistered multi-seed validation replication; it does
not authorize test access.  Test evaluation requires a later, independent,
reviewed promotion step after the architecture and selector are frozen.
