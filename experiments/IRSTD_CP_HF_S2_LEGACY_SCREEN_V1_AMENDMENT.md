# IRSTD-1K CP-HF-S2 V1 legacy-screen amendment

Status: preregistered before any CP-HF-S2 legacy-screen result is produced.

This amendment adds CP-HF-S2 as a second, independent candidate to the
existing model-design legacy screen. It does not edit, supersede, or change
the hashes of `IRSTD_PSBFR_V1_PROTOCOL.md` or its rules. CP-HF-S2 and PSBFR
remain separate graphs and separate run transactions.

## Fixed scope

This stage is a development-validation screen, not final confirmation. Its
maximum registry contains exactly three paired trajectories per route:

```text
(architecture_seed=42, run_seed=42)
(architecture_seed=42, run_seed=1446202191)
(architecture_seed=42, run_seed=104728269)
```

Every arm retains a 1000-epoch schedule identity and is operationally paused
only after the atomic epoch-500 commit. Records after epoch 500 cannot enter
the screen. Passing does not authorize a stability claim or public-test use.
The CP-HF-S2 runner enforces this internally: the request to enter epoch 501
raises a private control signal, re-reads the exact committed epoch-500
resume/history/candidate frontier, and exits successfully only after emitting
the fixed `EPOCH500_CLEAN` marker. A resume from epoch 500 repeats the same
validation and clean pause without executing epoch 501. Smoke runs do not
install this barrier.

The fixed arms are:

1. clean EviSIRST control;
2. frozen PSBFR V1 candidate;
3. `cp_hf_s2_v1` candidate.

The two candidates are compared independently with their same-seed clean
control. The gate never ranks PSBFR against CP-HF-S2 and never selects the
better observed candidate.
If both candidate rows are `GO`, both remain independent survivors. This gate
still names no winner; a new preregistered amendment must define any later
confirmation selection before either survivor's confirmation result is read.

## Sequential S1-to-S2 route contract

The three registered trajectories are not launched together. S1 first runs
only `(architecture_seed=42, run_seed=42)` for each route. Each route receives
its own immutable Seed-42 ledger and is decided independently:

```text
GO iff delta_mIoU > 0
       and not (delta_Pd < -0.003 and delta_Fa > 0)
       and that route's fixed mechanism diagnostic passes.
```

S1 has no mean-delta condition, ranks no candidates, and cannot name a winner.
A route with `STOP` is a preregistered futility stop and must not run seeds
`1446202191` or `104728269`. A route with `GO` authorizes only those two
remaining registered seeds for that same route. One route being missing,
waiting, or stopped never blocks the other route's S1 ledger or authorization.
The fixed S1 outputs are:

```text
runs/irstd_model_design/cp_hf_s2_v1/legacy_screen/comparison/
  stage1/<candidate>/seed42_interim.json
```

S2 is route-local. Only an S1-GO route may form a three-seed terminal ledger,
where the all-three-positive and mean-delta-at-least-0.002 rules below apply.
An S1-STOP route is terminal at S1 and is never treated as incomplete merely
because its prohibited remaining seeds do not exist. This sequential futility
screen makes the eventual evidence conditional on passing Seed 42; it is
development-screen evidence, not an unbiased stability estimate. Final five-
pair confirmation remains a separate preregistered stage.

## CP-HF-S2 architecture and training contract

- Dataset/target: `IRSTD-1K`, binary masks.
- Split: repository `splits/v2/IRSTD-1K`, split seed `20260811`, exactly 640
  train and 160 validation samples.
- Architecture API: only `experiments.irstd_cp_hf_s2_v1`; importing the old
  `EviSIRST_HF_Decoder_V2_reference` package is forbidden.
- Builder: `build_irstd_cp_hf_s2_v1(dataset="IRSTD-1K", seed=42,
  training=True)`.
- Validator/manifest authority: `validate_irstd_cp_hf_s2_v1`.
- Architecture key/label: `cp_hf_s2_v1` /
  `EviSIRST-CP-HF-S2-v1`.
- Full model is initialized from scratch. The 564 shared clean tensors must
  initially be bitwise identical to the clean Seed-42 constructor. Extension
  initialization must use an isolated deterministic substream.
- All base and extension parameters are trainable. The adapter must execute
  after the real `d2` decoder stage and before both `gt2` and `up_decoder1`.
- Adam, one parameter group, batch size 16, base LR `1e-3`, minimum LR
  `1e-5`, 10 warm-up epochs, validation every epoch, clean-R1 crop and
  augmentation, and the unchanged sum of six equally weighted BCE terms.
- Checkpoint selection uses the repository strict zero-margin selector:
  primary `best_mIoU`, secondary operating point `best_Pd`, no tolerance
  window.
- Public test data is unsupported. Every run/artifact records
  `test_split_accessed=false`.

The landed architecture source is the authority for source SHA-256, exact
state-key count, exact parameter count, and extension-key count. The runner
measures and binds these before GPU work; this amendment does not guess them
before that file exists.

CP-HF-S2 output is fixed at:

```text
runs/irstd_model_design/cp_hf_s2_v1/legacy_screen/formal/
  IRSTD-1K/binary/run_seed_<seed>/
```

No CLI output-root override is allowed.

Pre-freeze engineering produced multiple explicit CPU smoke transactions.
They are routed to content-addressed directories below
`.../legacy_screen/smoke/`, all record `promotion_eligible=false`, and the
entire smoke tree is development-only: it is excluded from S1/S2 evidence,
selection, mechanism diagnostics, comparison, and launch authorization.
These artifacts are retained for audit and are neither deleted nor promoted.
After this source freeze no new smoke transaction is authorized. Formal mode
remains locked to 1000 epochs and logical `cuda:0`; the frozen Stage-1 watcher
is the only authorized launcher and ignores the entire smoke tree. Direct use
of either route's training CLI, including a continuation seed, conveys no
authorization. The smoke CLI contract still requires both
`--smoke-max-train-samples` and `--smoke-max-val-samples`; supplying only one
cap is invalid.

A very small smoke validation subset may contain no target with area at most
9 pixels, for which the evaluator returns `tiny_pd=null`. Only in smoke mode,
and only when `tiny_target_count=0`, the runner records an explicit audit flag
and substitutes `tiny_pd=0.0` so the complete zero-margin key can exercise the
transaction. Formal metrics are never imputed.

## Source and split seals

Each run identity binds SHA-256 for the runner, this amendment, its JSON
rules, the architecture source, zero-margin selector, transaction engine,
and inherited clean-R1 dependencies. It also binds:

```text
manifest.json  5eeacbab52d8b70b44ce4cb70dfeda94cd326767f0c379fa96ab836c4c897596
train.txt      460083baae2ba23f5629e7bd346b5623f78a96856c83693b96bf917f6537ed2d
val.txt        05a0d0ecdb1772447c4b5b0b04a5e8cf3a748ba5fdab5cdb3f9323d9089e9576
data tree      ceec8eaca922fddb327b3091432d976d92aaeb45541136d43d3f754c3a1b2c30
```

Path escape, symlink, missing source, hash mismatch, or identity change fails
closed before launch/resume.

Every CP-HF-S2 training, candidate, history, checkpoint, summary, preflight,
diagnostic, and comparison payload explicitly records
`lockbox_accessed=false`, `test_selected=false`,
`test_selection_supported=false`, and `test_split_accessed=false` (and the
applicable public-test support/permission field is also false). Omission is a
contract failure, not an implicit false value.

## Strict transaction and no-clobber

`last_training_state.pth.tar` is the only resumable state. Strict resume
requires exact equality with a freshly rebuilt run identity plus model,
optimizer, captured CPU/CUDA/data-order RNG state, training/validation
histories, and the candidate ledger. The deterministic learning-rate schedule
is rebuilt from the frozen epoch and run identity rather than serialized as a
mutable scheduler. Strict model loading and architecture validation are
repeated after restore.

Each retained candidate is a unique
`candidates/epoch_NNNN.pth.tar`. It binds epoch, complete validation record,
zero-margin retention provenance, state contract, source/split hashes, and
`test_split_accessed=false`. Candidate paths are created exclusively; an
existing byte-identical artifact may be verified but never overwritten.
Candidate and checkpoint creation uses a held-parent-dirfd exclusive hard-link
commit; an existing destination is never replaced. JSON ledgers use the same
`O_DIRECTORY|O_NOFOLLOW` component walk and held-dirfd commit, with explicit
byte equality as the only idempotent-existing case.

The epoch-500 screen state is valid only after its resume payload, histories,
and all referenced candidates have atomically committed and revalidated. The
1000-epoch schedule is not rewritten to 500. Final `best_mIoU`/`best_Pd`
checkpoints are produced only if/when a 1000-epoch transaction is later
completed. Existing candidates, finals, summaries, and decision ledgers are
never silently replaced.

At every formal entry, before delegating to the inherited transaction engine,
the Stage-1 runner rejects any existing summary, either dual-role final, the
generic legacy final, and any symlink at those paths. If a latest state exists,
its safely loaded epoch must be in `[1, 500]`; epochs 501 through 1000 are
rejected even if an inherited finalize/resume path would otherwise skip the
dataset epoch hook. Thus a pre-existing completion artifact cannot bypass the
internal epoch-500 barrier.
The same entry check is repeated while holding the inherited per-run process
lock. The actual strict resume return must have `start_epoch<=501`, and the
inherited completed-run finalizer is unconditionally disabled for formal S1
(while remaining available to smoke transactions).

## Paired first-500 S2 gate

For each seed and each candidate, both control and candidate histories are
truncated to epochs 1 through 500 and independently reselected with the same
zero-margin complete key. Define

```text
delta(seed, candidate) = selected_mIoU(candidate) - selected_mIoU(clean)
```

Each S1-GO route receives its own S2 decision. Its metric screen passes only if:

```text
all three paired deltas > 0
mean of the three paired deltas >= 0.002
no seed has (delta_Pd < -0.003 and delta_Fa > 0)
```

PSBFR must additionally pass every mechanism invariant already frozen in its
own protocol/rules. CP-HF-S2 must demonstrate that its adapter executed,
departed from the identity state, and remained finite; these are mechanism
validity checks, not extra performance tuning knobs.

Those CP-HF-S2 checks come from one preregistered full validation pass over
each seed's exact first-500 zero-margin-selected checkpoint. A forward hook on
the installed adapter records its actual per-sample correction, verifies that
the hook ran for every validation sample, that the correction is finite and
nonzero, and that the elementwise absolute 0.25 feature-correction bound is
never exceeded. A learned nonzero `raw_scale` alone is not execution evidence.
The immutable diagnostic is selection-disabled and may not tune a threshold.

The comparison candidate registry is exactly
`["psbfr_v1", "cp_hf_s2_v1"]`. Route ledgers remain independent. A missing
arm for an S1-GO route is `INCOMPLETE`; an S1-STOP route requires no prohibited
S2 arms. Any additional candidate makes the ledger
`INVALID_MULTIPLE_CANDIDATES` unless a new amendment was frozen before any
result. Post-hoc seed choice, candidate choice, threshold choice, or candidate
ranking is forbidden.

The combined S2 path is reserved but its writer is sealed in this amendment:

```text
runs/irstd_model_design/cp_hf_s2_v1/legacy_screen/
  comparison/first500_clean_psbfr_cp_hf_s2_v1.json
```

This amendment implements only S1 authorization. It does not implement or
authorize route-local S2 terminal writing. S2 must arrive as a new additive
gate and addendum after S1 is terminal; it must consume the immutable S1 and
per-seed diagnostic evidence without modifying this runner, shared reader,
protocol, rules, or diagnostic sources (whose hashes are embedded in S1
evidence). The reserved combined path must remain absent.

## Later confirmation boundary

Five independently initialized paired 1000-epoch confirmation runs are a
separate stage. They may be implemented or launched only after both legacy
screens are terminal and the surviving architecture is frozen. This
amendment neither implements nor authorizes that confirmation or public-test
evaluation.
