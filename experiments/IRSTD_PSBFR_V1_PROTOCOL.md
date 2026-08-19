# IRSTD-1K PSBFR V1 architecture and validation protocol

Status: preregistered before any PSBFR validation result is produced.

This protocol defines the first model-design experiment after the formal
Seed-42 rejection of HF-Decoder V1.  Its objective is not to find one lucky
checkpoint.  Its objective is to decide whether a new inference graph can
reliably outperform the clean EviSIRST control under paired randomness and a
test-isolated evaluation protocol.

## 1. Closed evidence and problem statement

The following completed validation-only results are fixed evidence:

- clean EviSIRST, architecture/run seed 42: zero-margin best mIoU
  `0.6991869918699187` at epoch `411`;
- HF-Decoder V1, architecture/run seed 42: zero-margin best mIoU
  `0.6899375975039002` at epoch `402`;
- paired Seed-42 HF delta: `-0.0092493943660185`;
- clean EviSIRST, architecture seed 42 and run seed `1446202191`:
  zero-margin best mIoU `0.6969887569777499` at epoch `570`;
- HF-Decoder V1 under the same historical run seed: zero-margin best mIoU
  `0.6994492525570417` at epoch `490`.

HF-Decoder V1 therefore changes sign across the two observed trajectories.
Its correction-to-feature RMS ratio also differs by about eight-fold, and its
background gate response is larger than its target gate response in both
diagnosed checkpoints.  The failure is attributed to an unbounded
feature-space residual and direct branch/backbone co-adaptation.  HF-Decoder
V1 is closed and is not an eligible final model.

The development split used by those runs has been inspected repeatedly.  It
is henceforth named `legacy_dev_val`; it may screen a frozen candidate but may
not establish the paper's final stability claim.

## 2. Primary architecture: PSBFR V1

PSBFR means **Prediction-Supported Bounded Frequency Reconciler**.  Let
`F` be the final decoder feature and `z0` the clean final logit:

```text
F  = up_decoder1(d2, x1)
z0 = outc(F)
F_ro = stop_gradient(F)
L  = reflect_avg_pool_5x5(F_ro)
H  = F_ro - L
r  = Conv1x1_16_to_1(GELU(GN_1(Conv1x1_64_to_16([L, abs(H)]))))
S  = max_pool_7x7(sigmoid(stop_gradient(z0)))
z  = z0 + 2.0 * S * tanh(r)
```

Frozen constants:

- decoder channels: `32`;
- low-pass kernel: `5` with reflect padding;
- prediction-support kernel: `7`, stride `1`, padding `3`;
- hidden channels: `16`;
- normalization: `GroupNorm(1, 16)`;
- activation: `GELU`;
- correction bound: `delta_max=2.0`;
- corrector input: `[L, abs(H)]`; `S` is only an external envelope;
- first `1x1` convolution has no bias;
- terminal `1x1` convolution has a bias and both its weight and bias are
  initialized to exact zero;
- no learnable scalar gamma, support head, center head, support loss, Dice
  loss, loss reweighting, complete-target crop, or threshold search;
- all 564 base parameters and all five new state tensors are trainable;
- the base model is trained from scratch, not warm-started.

The formal graph contains 569 state tensors and 10,871,203 parameters.  At
construction `r=0`, hence `z` is bitwise equal to `z0`.  The terminal layer
receives a gradient on the first optimizer step; earlier corrector layers
begin receiving gradients after that terminal becomes nonzero.  Detaching
both `F` and `z0` makes the corrector a read-only evidence branch: it cannot
send its Jacobian directly into the backbone.

The spatial and amplitude trust-region invariant is:

```text
abs(z - z0) <= 2.0 * S <= 2.0
```

Because sigmoid support is soft, remote background is strongly attenuated,
not mathematically unchanged.  High-confidence true targets and false target
hypotheses can both receive positive or negative corrections.  The method is
not claimed to recover targets to which the base model assigns no support.

The existing forward API is preserved.  In training mode the model returns
the same six probability heads; corrected `z` is the sixth head and is also
fed into `d0`.  In evaluation mode the model returns only `sigmoid(z)`.  The
loss remains the unmodified sum of six probability-domain BCE terms.

## 3. D0 diagnostic, not the paper method

The current integrated EviSIRST reconstructs each encoder level as
`reconstruct(encoded) + 2 * skip`.  D0 changes only this to
`reconstruct(encoded) + skip`.  It adds no state tensors or parameters.

D0 is a diagnostic for accidental residual over-amplification.  It is not the
paper's primary novelty.  Two trajectories use the existing
clean controls with `(architecture_seed, run_seed)` equal to:

```text
(42, 42)
(42, 1446202191)
```

Every run is configured with the same 1000-epoch cosine schedule as its clean
control and is operationally paused immediately after the atomic epoch-500
commit.  It must not be configured as a 500-epoch schedule.  D0 becomes an
eligible PSBFR base only if, before any D0 result is inspected:

```text
both paired delta_mIoU values > 0
mean paired delta_mIoU >= 0.002
no seed has (delta_Pd < -0.003 and delta_Fa > 0)
```

Otherwise D0 is rejected and PSBFR remains attached to the clean graph.
PSBFR-on-clean may run concurrently with D0, but D0 and PSBFR must not be
combined until this gate is terminal.

## 4. Legacy-development screening

PSBFR V1 is screened at the atomic epoch-500 boundary on exactly three paired
trajectories:

```text
(architecture_seed=42, run_seed=42)
(architecture_seed=42, run_seed=1446202191)
(architecture_seed=42, run_seed=104728269)
```

Each pair uses identical data order, augmentation, optimizer, scheduler,
normalization, six-head BCE, batch size, worker count, and checkpoint
selection.  All variants retain the 1000-epoch total schedule in their run
identity and are operationally paused only after epoch 500 has committed.
Existing clean histories may be used only when their identity is an exact
match and they contain all 500 required epochs.  The selector is the
frozen zero-margin validation selector.  No result after epoch 500 may enter
the screening decision.

PSBFR advances only if all of the following were frozen before output:

```text
all three paired delta_mIoU values > 0
mean paired delta_mIoU >= 0.002
no seed has (delta_Pd < -0.003 and delta_Fa > 0)
the recorded bound abs(delta_logit) <= 2*S holds everywhere
the corrector is not identically zero
the absolute tanh saturation fraction at |tanh(r)| >= 0.99 is < 0.10
```

Failure is terminal for PSBFR V1.  Kernel sizes, width, `delta_max`, support
definition, loss, and threshold must not be tuned after this decision.  A new
idea would require a new version and a new protocol.

## 5. Locked confirmation and stability

Before any PSBFR validation metric is produced, a new `confirm_lockbox` must
be deterministically carved from the 640 current training members and frozen
as an immutable manifest.  Those images were historically part of the
training pool, so the paper must call this a **new locked confirmation set
from the original training pool**, not a pristine external test.  Neither
baseline nor PSBFR may train on those members in confirmation runs.

After the 500-epoch gate, clean EviSIRST and the frozen final PSBFR graph are
trained from scratch for 1000 epochs on five paired, independently initialized
trajectories.  The public seed derivation is:

```text
seed(kind, i) = uint32_be(SHA256("EviSIRST/PSBFR-v1/" + kind + "/" + i)[0:4]) mod 2^31
```

The resulting `(init_seed, run_seed)` registry is:

```text
i=0: (1186821503, 1442745291)
i=1: (1664584613, 560126688)
i=2: (984034075, 1740114274)
i=3: (518560408, 2076447678)
i=4: (432975590, 922968363)
```

The current public `initialize_evisirst` constructor deliberately accepts
only architecture seed 42.  It is sufficient for the legacy screen but not
for this confirmation.  Before confirmation begins, a separate hash-bound
variable-initialization constructor must demonstrate bitwise parity with the
public constructor at seed 42, deterministic reproducibility at every allowed
seed, different shared-base initialization across different seeds, caller-RNG
isolation, and exact restoration of any temporarily adapted builder state.
Changing only runtime seeds while leaving initialization fixed does not meet
the independent-initialization requirement.

Within every pair, the 564 shared base tensors are initialized bitwise
identically.  PSBFR extension initialization is derived deterministically
from the pair's `init_seed` as
`uint64_be(SHA256("EviSIRST/PSBFR-v1/extension/" + init_seed)[0:8]) mod 2^63`,
in an isolated RNG substream, and does not consume the runtime RNG.
Checkpoints bind the base init seed, derived extension seed, and runtime seed
separately.

The final paper may use the phrase **stably outperforms the clean baseline**
only when all conditions hold on the five paired lockbox evaluations:

```text
mean paired delta_mIoU >= 0.002
all 5 paired delta_mIoU values > 0
the lower bound of the two-sided seed-level paired 95% t interval > 0
no seed has (delta_Pd < -0.003 and delta_Fa > 0)
```

Seed-level inference is primary.  Paired image bootstrap may describe sample
uncertainty but cannot replace seed variation.  If fewer than five deltas are
positive, the wording must be weakened to the observed positive-seed count.

## 6. Required controls and diagnostics

The minimum controlled table is:

1. clean EviSIRST;
2. D0-only;
3. PSBFR-only;
4. D0+PSBFR only if D0 passed its frozen gate;
5. parameter-matched bounded reconciler with `S=1`;
6. parameter-matched reconciler using ordinary final features instead of
   `[L, abs(H)]`;
7. HF-Decoder V1 as the unbounded feature-space negative control.

The paired promotion baseline is the immediate 564-key clean EviSIRST parent,
which is the stricter causal control for the added module.  The paper must
also retrain upstream SCTransNet from scratch under the same frozen data,
selector, seed registry, and budget; historical test-selected SCTransNet
numbers are not an eligible comparator.  A final claim of outperforming
"the baseline" must name which of these two baselines it means and report
both in the main results table.

The main comparison uses identical training recipe.  Training-recipe changes
such as complete-target crop must appear in a separate 2x2
architecture-by-recipe table and cannot establish the structural claim.

Every run records mIoU, nIoU, precision, recall, F1, Pd, Fa, tiny-target Pd,
false objects per image, matched-target recall, matched component area ratio,
centroid error, target/background/false-component support, signed logit
correction, correction energy, and tanh saturation.  Parameter count, MACs,
peak memory, and batch-1 median/p95 latency are measured on the same hardware.

## 7. Publication scope and prior-art boundary

PSBFR does not claim the first use of frequency information, dynamic filters,
coarse-to-fine refinement, noise suppression, or logit-domain learning in
infrared small-target detection.  Closest-risk directions include DHiF,
NS-FPN, scale/location-sensitive objectives, target-level posterior modeling,
and generic patch/logit refinement.

The defensible hypothesis is narrower:

> A base-prediction support envelope defines a spatial trust region, while a
> signed and explicitly bounded output-logit residual reconciles local
> frequency evidence without directly back-propagating the correction
> Jacobian into the backbone.

Novelty remains `needs-literature-search` until a full closest-work review is
frozen.  Results remain `TBD` until executed; this protocol contains no
expected or invented performance values.

## 8. Test isolation and stopping rules

- `legacy_dev_val` may select checkpoints and screen the architecture.
- `confirm_lockbox` is opened once per frozen 1000-epoch paired checkpoint.
- Public test data is unavailable to training, checkpoint selection, module
  selection, hyperparameter selection, calibration, and thresholds.
- Final benchmark evaluation is a later locked post-development evaluation,
  never a feedback loop.
- Any non-finite value, identity mismatch, missing transaction artifact,
  resume mismatch, test-access flag, or violation of the correction bound is
  fail-closed.
- No claim of success is written before the corresponding immutable decision
  ledger is produced and freshly revalidated.
