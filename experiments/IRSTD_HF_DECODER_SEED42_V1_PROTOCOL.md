# IRSTD-1K HF Decoder V1 fixed-Seed-42 formal revalidation

Status: preregistered, validation-only, single-seed formal revalidation.

## Purpose

This run freezes runtime seed 42 as the sole formal training seed for the
IRSTD-1K HF-Decoder V1 model.  It is a clean re-run of the already frozen
572-state-tensor architecture; it is not a new architecture search, a
multi-seed study, or permission to choose whichever seed performs best.

The completed `run_seed=1446202191` experiment remains a historical pilot.
Its validation result may be disclosed as provenance, but it is not a pool
from which the better seed may be selected.  Once this protocol is started,
the Seed-42 result is the formal single-seed result whether it is higher or
lower than the historical pilot.

## Immutable run contract

- Repository: `/home/ly/EviSIRST_main`.
- Dataset and target: `IRSTD-1K`, binary masks.
- Split: frozen repository V2 manifest with split seed `20260811`, exactly
  640 training and 160 validation samples.  Sample-level fallback must be
  acknowledged explicitly.
- Architecture seed: `42`.
- Runtime/training seed: `42`.
- Architecture: the frozen `IRSTD-HF-Decoder-v1` graph from
  `experiments/irstd_hf_decoder_v1.py`: 564 base state tensors plus 8 HF
  state tensors, 572 total.  No architecture file is modified by this run.
- Initialization: full model from scratch.  The HF residual starts as an
  exact identity and every base and HF parameter is trainable.
- Optimization: one Adam parameter group; 1000 epochs; batch size 16; base
  learning rate `1e-3`; minimum learning rate `1e-5`; 10 warm-up epochs;
  validation every epoch.
- Training semantics: unchanged clean-R1 crop/augmentation and the sum of six
  equally weighted BCE probability-head losses.  No complete-target crop,
  loss modification, center head, warm start, or public-test feedback.
- Selection: validation only with the frozen strict complete key and zero
  margin.  Both `best_mIoU` and `best_Pd` roles must be retained as distinct
  final checkpoints.  No `0.001` window is allowed.
- Output: isolated under
  `runs/irstd_performance/hf_decoder_seed42_v1/formal/IRSTD-1K/binary/run_seed_42/`.
- Test: unavailable.  The runner must not import or construct a test dataset,
  test loader, public-test evaluator, or public-test checkpoint selector.

## Integrity and recovery contract

The source identity must hash this wrapper, this protocol, the frozen HF
runner, the frozen HF architecture, the zero-margin selector, and all source
dependencies inherited from the validation-selected training engine.  The
Seed-42 schemas and output tree are distinct from the historical Seed-144 run.

`last_training_state.pth.tar` is the only resumable transaction state.  `--resume` must
restore model, Adam, RNG, sampler/generator, validation history, candidate
artifacts, and HF identity state under the same immutable run identity.  A
completed transaction must be finalized crash-idempotently into:

- `EviSIRST_best_mIoU.pth.tar`
- `EviSIRST_best_Pd.pth.tar`

Each final must bind its exact selected epoch, complete selection key, source
candidate SHA-256, and 572-tensor state.  Re-running finalization must either
reproduce the same evidence or fail closed on any mismatch.

## Interpretation

This experiment produces the formal single-Seed-42 validation result.  It may
be compared descriptively with the historical Seed-144 pilot, but the seed is
not selected after observing the two outcomes.  Because no Seed-42 clean or
complete-target paired control is part of this run, this revalidation alone
does not establish a Seed-42 causal HF gain over those controls.  Public-test
evaluation remains a later, separately authorized step after the result and
code are frozen.
