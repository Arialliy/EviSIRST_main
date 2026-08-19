# IRSTD DCS-PG full-train test-selected V1 protocol

Status: **implementation lock; this document alone does not authorize a GPU
launch**.

## Two evidence layers must remain separate

Layer A is the existing 640-train/160-validation design screen.  It is the
place for validation-only structure decisions, paired multi-seed stability,
and later confirmatory lockbox work.  This runner does not implement or access
Layer A's validation role.

Layer B, defined here, reproduces the historical IRSTD-1K full-train
test-selected convention.  The frozen source train set has all 800 samples;
the former 160 validation members rejoin optimization and are not evaluated as
an independent split.  This prevents a false comparison between a 640-image
candidate and the historical 800-image baseline.

## Registered arms and initial authorization

The runner registers six independent method IDs:

| Method ID | Architecture | Training loss | Initial formal status |
| --- | --- | --- | --- |
| `clean_original` | clean EviSIRST | original BCE | blocked control |
| `clean_farbg` | clean EviSIRST | BCE + far-background | blocked control |
| `cp_parent_original` | CP-HF-S2 | original BCE | blocked control |
| `cp_parent_farbg` | CP-HF-S2 | BCE + far-background | blocked control |
| `dcspg_original` | CP-HF-S2 + DCS-PG | original BCE | authorized |
| `dcspg_farbg` | CP-HF-S2 + DCS-PG | BCE + far-background | authorized |

Every arm is a separate scratch run with its own identity, recovery directory,
history, candidates, selection, summary, and exactly two formal weights.  No
state or optimizer moment is shared.  The four blocked controls require a
separate authorization refreeze; until run, this package cannot support a
complete six-arm ablation claim.

## Evidence boundary

The official test split deliberately selects checkpoints to match the
historical baseline convention.  Every artifact committed after first test
access states `test_selected=true`, `selection_is_optimistic=true`,
`unbiased_test_claim_supported=false`, and
`stable_over_baseline_claim_supported=false`.

The result is only like-for-like exploratory evidence against a historical
test-selected baseline.  It is not untouched, unbiased, confirmatory, or proof
of stable superiority.  Test results cannot choose architecture, loss recipe,
lambda, dilation, or another hyperparameter.

## Fixed full-train run per arm

- Dataset: `IRSTD-1K`; architecture/run seed 42.
- Optimization: the standard frozen 800-sample source train index in original
  order, exactly equal to the V2 split manifest's train/validation union.
- Training loader/augmentation/target: the historical
  `EviSIRSTTrainDataset` protocol (`raw_mask/255`, patch 256), matching the
  legacy test-selected baseline runner.
- Official test: 201 samples, legacy normalization.
- Exactly 1000 epochs, batch size 16, workers 0.
- One Adam group, base LR `1e-3`, minimum LR `1e-5`, warm-up 10 epochs.
- Full scratch; no baseline, design-screen, previous-arm checkpoint, or
  optimizer is loaded.
- Common evaluator V1: probability `>0.5`, 8-connected objects, Hungarian
  one-to-one matching, distance `<3`, tiny area `<=9`.

## Genuine lazy test access

Epochs 1--500 are training-only.  Startup and these epochs must not construct
`EviSIRSTTestDataset` or open a test index, image, or mask.  The immutable
identity binds only preregistered expected test hashes from the rules and
records `test_index_opened_at_identity_construction=false` and
`startup_test_index_opened=false`. These immutable startup facts are not
reused as a claim about later epochs.

Immediately before the epoch-501 transaction can construct or open test data,
it no-clobber commits `test_access_started.json`. This pre-access ledger binds
the expected test contract, epoch 501, the run identity, and the optimistic
evidence boundary without itself reading test. After live hash verification,
`test_access_verified.json` binds the observed identity. Thus a crash after
opening test but before committing epoch 501 cannot make an epoch-500 recovery
look test-blind: resume sees the started ledger and conservatively treats test
access as started, verifies the live split again, and continues epoch 501.

Test then runs after every epoch 501--1000, yielding exactly 500 ordered
records. An epoch-500 recovery committed before the started ledger has
`test_history=[]`, `test_split_accessed=false`, `test_selected=false`, and
`selection_is_optimistic=false`; epoch 501 onward has true flags. Resume at or
before epoch 500 remains test-blind only when the started ledger is absent;
resume after epoch 500 requires both access ledgers and verifies test before
continuing.

## Dual formal weights

The test history alone is replayed with strict lexicographic keys.  Exact ties
retain the earlier epoch through `-epoch`:

```text
best_miou = (mIoU, Pd, -Fa, nIoU, tiny_Pd, -loss, -epoch)
best_pd   = (Pd, -Fa, tiny_Pd, mIoU, nIoU, -loss, -epoch)
```

Each arm publishes exactly:

```text
published_weights/EviSIRST_best_mIoU.pth.tar
published_weights/EviSIRST_best_Pd.pth.tar
```

Metrics from different roles must never be combined into a synthetic vector.

## Frozen far-background loss

Original arms use the existing sum of six mean BCE terms bit-for-bit.

For a binary training mask, `G=(mask>0.5)`.  Its protected region is binary
dilation by the 29-point Euclidean disk in a 7x7 window,
`u^2+v^2<=9`—not square max-pooling.  `F` is its complement.  Only the training
tuple's raw `d0` and final `out` probability heads (`[-2]`, `[-1]`) enter:

```text
a_h = F * (relu(p_h - 0.5) / 0.5)^2
K = min(9, number_of_far_background_pixels)
L_h = mean(topK(a_h, K)) per sample, or 0 when K=0
L_FA = batch_mean((L_d0 + L_out) / 2)
loss = original_six_head_BCE_sum + 1.0 * L_FA
```

Constants are tied to declared protocol constants: threshold 0.5, match radius
3, tiny-area cutoff 9, and one unit weight for one averaged auxiliary term.  No
sweep is permitted.  Validation/test masks never enter the loss.  Training
history records `base_bce_sum`, `farbg_loss`, and `total_loss` separately.

## Identity, transaction, and resume

The identity binds method ID, architecture and tests, executable dependencies,
protocol/rules hashes, V2 source-train manifest/tree, standard 800-sample train
index/order, preregistered test hashes, normalization, loss recipe, evaluator,
schedule, selector, seeds, and output path.

The CP-HF-S2 parent dependency is independently pinned to SHA-256
`0dba0021dfe8f46d8c4e2932b478ce03609ab14be73b20c8b77817dce240436a`;
the guard module's own dependency declaration is not trusted as the sole
source of this value.

The source manifest must also include the frozen model-builder dependency
`experiments/four_dataset_models_seed42_v1.py` and this runner's CPU contract
test `tests/test_train_irstd_dcspg_ablation_test_selected_v1.py`. Their file
digests participate in the aggregate source digest, so either content change
changes the immutable run identity.

One epoch consists of training, optional lazy test activation/evaluation,
candidate updates, and an atomic recovery commit.  Resume accepts only the
latest strictly validated commit.  Model/Adam tensors are finite; histories,
role epochs, candidate SHA-256 values, and identities replay exactly.
Duplicate JSON keys, NaN/Infinity, symlinks, cross-arm artifacts, unknown
files/temporaries, cadence changes, and overwrite attempts fail closed.

Filesystem path checks reject symlinks in the complete existing ancestor chain
and require the canonical run directory to remain strictly inside the output
root. The residual check-to-open TOCTOU threat assumes the declared
single-writer run lock and a non-malicious host administrator; defending
against concurrent privileged path replacement is outside this V1 scope.

Finalization no-clobber writes and tensor-reopens both weights, records their
metrics/role keys/SHA-256 in `selection_record.json`, and commits `summary.json`
last.  Resume accepts an existing final only if its complete payload replays.

## Required completed artifacts per arm

- `last_training_state.pth.tar`;
- `training_history.json`, epochs 1--1000;
- `test_history.json`, epochs 501--1000;
- `test_access_started.json` and `test_access_verified.json`;
- `selection_record.json` and `summary.json`;
- exactly two regular files under `published_weights/`.

No experimental result is asserted here.
