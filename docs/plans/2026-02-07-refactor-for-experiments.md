# OAR Codebase Refactoring Plan

> **For Agent:** REQUIRED: Use the executing-plans skill to implement this plan task-by-task.

**Goal:** Refactor the codebase to separate the OAR framework from MNIST-specific code, enabling easy addition of new experiments (e.g., Penn Treebank).

**Architecture:** Extract reusable OAR components (regularizers, activations, layers, quantizers) into an `oar/` package. Move experiment-specific code into `experiments/` with a shared base. Generalize the four-step quantization as configurable training steps.

**Tech Stack:** TensorFlow 2.10.1, QKeras 0.9.0, Python 3.10

---

## Deep Investigation Findings (Feb 7, 2026)

**QKeras Internal Architecture:**
- `QSimpleRNN` wraps `QSimpleRNNCell`, which extends Keras `SimpleRNNCell`
- Quantizers use internal naming: `self.kernel_quantizer` (original) vs `self.kernel_quantizer_internal` (resolved callable)
- STE pattern: `x + tf.stop_gradient(-x + xq)` - forward uses quantized, backward uses full precision
- Activations are quantizers: `"quantized_tanh"` goes through `get_quantizer()`
- `activity_regularizer` only sees post-activation output - NOT per-timestep, NOT pre-activation

**TensorFlow RNN Internals:**
- `activity_regularizer` is applied AFTER `call()` returns, via `Layer._handle_activity_regularization()`
- For RNNs: applied to final output only, NOT per-timestep
- Per-timestep regularization MUST be done inside `cell.call()` with `unroll=True`
- Pre-activations are only accessible inside `cell.call()` - this is why OAR is applied there

**Current Implementation - What Works:**
- `QSimpleRNNCellWithOAR.call()` correctly captures pre-activations (`h + h_2`) before activation
- OAR2 uses `add_loss()` which integrates correctly with Keras training loop
- `unroll=True` is forced when OAR is enabled (required for per-timestep `add_loss()`)
- Gradient scaling trick is correct: `h/s + tf.stop_gradient(-h/s + h)`

**Fragilities Identified (MUST FIX):**

| Issue | Location | Risk | Description |
|-------|----------|------|-------------|
| `preacts` shape | `model_utils.py:284-290` | **HIGH** | `[batch_size, units]` - breaks with different batch sizes at inference |
| Incomplete `get_config()` | `quantization.py:73-77` | **HIGH** | Missing `qnoise_factor`, `use_ste`, etc - model save/clone fails |
| `GeneralActivation.get_config()` | `model_utils.py:602-605` | **HIGH** | References non-existent `self.alpha_init` |
| Lambda with captured vars | `mnist_rnn_model.py:45-58` | **MEDIUM** | Cannot serialize - use custom layer instead |
| Missing `nest` import | `model_utils.py:293` | **MEDIUM** | Relies on QKeras re-export, should be explicit |
| `return_sequences` ignored | `model_utils.py:78` | **LOW** | Hardcoded to `True`, kwarg silently ignored |

**Validation Confirmed:**
- OAR2 metric (via `tf.keras.metrics.Mean`) correctly averages across all timesteps
- Weight transfer between steps correctly resets stat variables
- Ternarization threshold `τ = t * E(|θ_l|)` computed correctly

---

## Current State

```
main.py                    # Entry point, training loops, four-step quantization (524 lines)
mnist_rnn_model.py         # MNIST model architecture (190 lines)
quantization.py            # TernarizationWithThreshold quantizer (77 lines)
utils/model_utils.py       # Everything else: layers, regularizers, activations (885 lines)
```

**Problems:**
- `model_utils.py` is an 885-line grab bag of unrelated components
- Layer names hardcoded (`QRNN_0`, `QRNN_1`, `DENSE_0`, `DENSE_OUT`)
- Step logic hardcoded (`if step == 2:`, `if step == 3:`, `if step == 4:`)
- No separation between framework and experiment
- Critical serialization bugs (see Fragilities above)
- Dead code: `custom_loader`, `sign_with_ste`

---

## Target State

```
oar/                           # Core framework (reusable across experiments)
├── __init__.py
├── regularizers.py            # OAR2, oar_penalty_fn, compute_oar_metric
├── activations.py             # sign_ste_tanh, mod_sign, signed_residue, TrackedActivation
├── layers.py                  # QRNNWithOAR, QSimpleRNNCellWithOAR, QDenseWithOAR, Downsampling
├── quantizers.py              # TernarizationWithThreshold, ternarize_tensor_with_threshold
├── training.py                # QuantizationStep dataclass, step execution logic
└── callbacks.py               # ReservoirHistogramCallback for preact visualization

experiments/
├── __init__.py
├── base.py                    # BaseExperiment class (optional)
└── mnist/
    ├── __init__.py
    ├── config.py              # MNIST_OPTIONS, MNIST_STEPS, layer definitions
    ├── model.py               # get_model() - MNIST-specific architecture
    └── data.py                # get_datasets(), normalize_img(), resize(), augment()

main.py                        # Simplified entry point - picks experiment, runs it
```

---

## Phase 0: Critical Bug Fixes

**Goal:** Fix the HIGH-risk issues that break serialization and inference.

**Changes:**

1. **Fix `TernarizationWithThreshold.get_config()`** - add all constructor parameters:
   - `qnoise_factor`, `var_name`, `use_ste`, `use_variables` (currently only `threshold`)

2. **Fix `GeneralActivation.get_config()`** - references non-existent `self.alpha_init`:
   - Should return `{"activation": self.activation, "name": self.name}`

3. **Fix `preacts` weight shape** - currently `[batch_size, units]`:
   - Change to scalar EMA per unit `[units]` using `tf.reduce_mean(tf.abs(...), axis=0)`
   - OR remove batch_size dependency entirely (Phase 1.5 reservoir sampling will replace this)

4. **Add explicit `nest` import** in `model_utils.py`:
   - `from tensorflow.python.util import nest`

5. **Replace Lambda layer** in `mnist_rnn_model.py` with custom `TernarizeInputLayer`:
   - Custom layer with proper `get_config()` for serialization

6. **Register custom objects globally** - add registration at module load:
   - All custom layers, quantizers, activations need `@tf.keras.utils.register_keras_serializable`

**Risk:** Medium - fixing bugs, but touching critical paths

**Validation:**
1. `model.save()` / `tf.keras.models.load_model()` round-trip works
2. `tf.keras.models.clone_model()` works
3. Run MNIST training, verify identical behavior

---

## Phase 1: Naming Convention Alignment + Cleanup

**Goal:** Consistent naming with paper, remove dead code.

**Naming Changes:**

| Current | New | Notes |
|---------|-----|-------|
| `oar_hat_fn` | `oar_penalty_fn` | Paper's "hat" penalty |
| `oar_hat_metric_fn` | `compute_oar_metric` | Fraction in valid range |
| `mod_sign_with_tanh_deriv` | `mod_sign` | ModSign (Eq.4) |
| `sign_with_tanh_deriv` | `sign_ste_tanh` | Sign with tanh gradient |
| `GeneralActivation` | `TrackedActivation` | Activation wrapper |
| `TimeReduction` | `Downsampling` | Keep alias for backwards compat |
| `ModelWithGradInfo` | `OARModel` | Optional gradient logging |
| `oar_bits`, `precision` | `omega` | Bit-width (k = 2^ω) |
| `oar_lm`, `lm` | `oar_lambda` | Regularization rate |

**Dead Code Removal:**
- `custom_loader` (lines 812-831) - never called
- `sign_with_ste` (lines 486-493) - never used

**Keep:**
- `OAR1` class - mentioned in paper as linear variant
- Commented gradient logging - add `log_gradients=False` flag instead

**Risk:** Low - renaming + removal of unused code

**Validation:**
1. All renamed functions produce identical outputs
2. MNIST training works identically

---

## Phase 1.5: Preactivation Histogram - Reservoir Sampling

**Goal:** Replace biased EMA tracking with unbiased reservoir sampling.

**Problem:** Current EMA (`0.9 * old + 0.1 * new`) is biased toward later timesteps/batches.

**Solution:** Vitter's Algorithm R for reservoir sampling.

**Changes:**
- Replace `preacts` EMA weight with `preact_reservoir` (shape `[100000]`) + `reservoir_count`
- Update reservoir every N batches (`RESERVOIR_UPDATE_EVERY`, default 20) to reduce overhead
- Use a NumPy-backed reservoir update via `tf.numpy_function` to avoid slow TF scatter ops in 2.10
- New `ReservoirHistogramCallback` logs histograms at epoch end

**Risk:** Low - only affects visualization, not metrics/training

**Validation:**
1. OAR metrics unchanged
2. Histograms statistically match full data distribution (account for batched updates)
3. Training throughput improves versus per-batch reservoir updates

---

## Phase 2: Extract `oar/` Package

**Goal:** Create reusable framework package.

**Structure:**
- `oar/regularizers.py` - `OAR2`, `oar_penalty_fn`, `compute_oar_metric`
- `oar/activations.py` - `sign_ste_tanh`, `mod_sign`, `signed_residue`, `TrackedActivation`
- `oar/layers.py` - `QRNNWithOAR`, `QSimpleRNNCellWithOAR`, `QDenseWithOAR`, `Downsampling`
- `oar/quantizers.py` - `TernarizationWithThreshold`, `ternarize_tensor_with_threshold`
- `oar/__init__.py` - clean public API

**Risk:** Medium - import path changes

**Validation:**
1. All imports resolve
2. MNIST training identical to baseline

---

## Phase 3: Create `experiments/mnist/`

**Goal:** Move MNIST-specific code to experiment directory.

**Structure:**
- `experiments/mnist/model.py` - from `mnist_rnn_model.py`
- `experiments/mnist/data.py` - `get_datasets()`, normalization, augmentation
- `experiments/mnist/config.py` - `MNIST_OPTIONS`, layer definitions

**Risk:** Medium - file reorganization

**Validation:**
1. Model architecture identical (same parameter counts)
2. Data loading identical
3. MNIST training identical

---

## Phase 3.5: Organized Experiment Outputs

**Goal:** Reorganize four-step quantization outputs so all steps of a run are grouped together with proper config/log files.

**Structure:**
```
runs/mnist/20260208-123456/
├── config.json                    # Initial experiment options
├── output.log                     # Full experiment log (all steps)
├── step_1/
│   ├── config.json                # Step-specific config (options + layer_options)
│   ├── checkpoints/
│   └── logs/tensorboard/
├── step_2/
│   └── ...
├── step_3/
│   └── ...
└── step_4/
    └── ...
```

**Note:** No per-step `output.log` - the experiment-level log captures all output. Search for "STEP N/4" markers to find step boundaries.

**Changes:**
- Add `create_run_dir()` function and `step`/`run_dir` parameters to `train()`
- Save `config.json` for each step (options + layer_options, with activation functions serialized to names)
- Add `tee_output()` context manager for experiment-level logging
- Update `perform_four_step_quant()` to create shared run directory
- Save initial experiment `config.json` before any mutations
- Extract `_make_oar_config()` helper (DRY fix)
- Use `verbose=2` for cleaner log files (one line per epoch)

**Risk:** Low - only changes output organization, not training logic

**Validation:**
1. All 72 tests pass
2. Training produces organized output structure
3. Config files are valid JSON with all options

**Detailed Plan:** See `docs/plans/2026-02-08-phase3.5-organized-outputs.md`

---

## Phase 4: Generalize Training Steps

**Goal:** Configuration-driven quantization steps.

**Create `QuantizationStep` dataclass:**
- `name`, `epochs`, `activation`, `gradient_scale`, `ternarize_inputs`, `quantize_weights`, `use_oar`, `learning_rate_multiplier`, `omega`

**Define `MNIST_STEPS`:**
- Step 1: tanh baseline
- Step 2: sign_ste_tanh + gradient scaling
- Step 3: + ternarize inputs
- Step 4: + quantize weights + OAR

**Risk:** High - core training logic

**Validation:**
1. Each step produces identical `layer_options`
2. Final accuracy within 0.1% of baseline

---

## Phase 5: Final Cleanup

**Goal:** Prepare for Penn Treebank.

**Changes:**
- Simplify `main.py` CLI
- Update `CLAUDE.md`
- Verify extensibility

**Risk:** Low

---

## Key Invariants (MUST PRESERVE)

1. **OAR applied before activation, to pre-activations** - inside `cell.call()`, NOT via `activity_regularizer`
2. **`unroll=True` when OAR enabled** - required for per-timestep `add_loss()`
3. **Gradient scaling**: `h/s + tf.stop_gradient(-h/s + h)` - forward unscaled, backward scaled
4. **OAR2 metric**: `tf.keras.metrics.Mean` correctly averages across timesteps
5. **Ternary threshold**: `τ = t * E(|θ_l|)` per layer
6. **ModSign + OAR coupling**: both use same `omega` (k = 2^ω)
7. **Reproducibility**: `tf.random.set_seed(1997)`

---

## QKeras API Reference (from source investigation)

**QSimpleRNNCell accepted parameters:**
- `units`, `activation` (can be quantized like `"quantized_tanh"`)
- `kernel_quantizer`, `recurrent_quantizer`, `bias_quantizer`, `state_quantizer`
- `kernel_regularizer`, `recurrent_regularizer`, `bias_regularizer`
- Standard Keras: `dropout`, `recurrent_dropout`, `use_bias`, initializers, constraints

**Quantizer interface:**
- Must implement `__call__(self, x)` returning quantized tensor
- Must implement `get_config()` returning all constructor args
- `from_config(cls, config)` defaults to `cls(**config)`
- STE: `x + tf.stop_gradient(qnoise_factor * (-x + xq))`

**Important QKeras patterns:**
- Two-phase storage: `kernel_quantizer` (original) vs `kernel_quantizer_internal` (resolved)
- Truthiness check: `if self.kernel_quantizer:` uses original, call uses `*_internal`

---

## Execution Order

| Phase | Risk | Dependencies | Status |
|-------|------|--------------|--------|
| 0     | Med  | None         | ✅ Done |
| 1     | Low  | Phase 0      | ✅ Done |
| 1.5   | Low  | Phase 1      | ✅ Done |
| 2     | Med  | Phase 1.5    | ✅ Done |
| 3     | Med  | Phase 2      | ✅ Done |
| 3.5   | Low  | Phase 3      | ✅ Done |
| 4     | High | Phase 3.5    | Pending |
| 5     | Low  | Phase 4      | Pending |

**Recommended:** Complete each phase, validate with test run before proceeding.
