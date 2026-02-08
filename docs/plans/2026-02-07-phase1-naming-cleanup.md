# Phase 1 Naming + Cleanup Implementation Plan

> **For Agent:** REQUIRED: Use the executing-plans skill to implement this plan task-by-task.

**Goal:** Align naming with the paper and remove dead code with no backward compatibility.

**Architecture:** Rename OAR functions/classes and option keys at the source, then update all call sites to the new names. Remove unused utilities and keep the behavior identical while changing the public surface.

**Tech Stack:** Python 3.10, TensorFlow 2.10.1, QKeras 0.9.0, pytest

---

### Task 1: Add Phase 1 naming tests

**Files:**
- Create: `tests/test_phase1_naming.py`

**Step 1: Write the failing test**

```python
import tensorflow as tf

from utils.model_utils import (
    compute_oar_metric,
    oar_penalty_fn,
    sign_ste_tanh,
    mod_sign,
    TrackedActivation,
)


def test_sign_ste_tanh_outputs_sign():
    x = tf.constant([-2.0, 0.0, 2.0])
    out = sign_ste_tanh(x)
    tf.debugging.assert_equal(out, tf.constant([-1.0, 0.0, 1.0]))


def test_mod_sign_outputs_signed_mod():
    x = tf.constant([-1.0, 1.0, 5.0])
    out = mod_sign(x, num_bits=3)
    tf.debugging.assert_equal(out, tf.constant([-1.0, 1.0, -1.0]))


def test_oar_penalty_is_zero_at_origin():
    x = tf.constant([0.0, 0.0, 0.0])
    out = oar_penalty_fn(x, k=8, a=1.0)
    tf.debugging.assert_equal(out, tf.zeros_like(out))


def test_compute_oar_metric_is_one_at_origin():
    x = tf.constant([[0.0, 0.0]])
    out = compute_oar_metric(x, k=8, a=1.0)
    tf.debugging.assert_near(out, tf.constant([1.0]))


def test_tracked_activation_applies_activation():
    layer = TrackedActivation(activation=tf.nn.relu)
    x = tf.constant([-1.0, 2.0])
    out = layer(x)
    tf.debugging.assert_equal(out, tf.constant([0.0, 2.0]))
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_phase1_naming.py -v`
Expected: FAIL with import errors for missing renamed symbols.

**Step 3: Commit**

```bash
git add tests/test_phase1_naming.py
git commit -m "test: add phase 1 naming expectations"
```

---

### Task 2: Rename OAR functions/classes and remove dead code

**Files:**
- Modify: `utils/model_utils.py`

**Step 1: Write the failing test**

The tests from Task 1 should still be failing before code changes.

**Step 2: Implement minimal code to pass**

```python
# Rename these functions/classes (no aliases, no old names kept):
# oar_hat_fn -> oar_penalty_fn
# oar_hat_metric_fn -> compute_oar_metric
# sign_with_tanh_deriv -> sign_ste_tanh
# mod_sign_with_tanh_deriv -> mod_sign
# GeneralActivation -> TrackedActivation
# TimeReduction -> Downsampling
# ModelWithGradInfo -> OARModel

# Remove dead code:
# - sign_with_ste
# - custom_loader

# Add log_gradients flag to OARModel, and keep existing gradient logging
# blocks gated behind it (leave them commented or toggleable).
```

**Step 3: Run tests to verify they pass**

Run: `pytest tests/test_phase1_naming.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add utils/model_utils.py
git commit -m "refactor: rename OAR utilities and remove dead code"
```

---

### Task 3: Update option keys and call sites (no backward compatibility)

**Files:**
- Modify: `main.py`
- Modify: `mnist_rnn_model.py`
- Modify: `utils/model_utils.py`

**Step 1: Write the failing test**

Add a small regression test in `tests/test_phase1_naming.py` to validate
the new option keys by calling a minimal `get_default_layer_options_from_options`
flow with `omega` and `oar_lambda` and asserting they are used.

```python
from utils.model_utils import get_default_layer_options_from_options


def test_default_layer_options_use_omega_and_oar_lambda():
    options = {
        "oar": {"omega": 6, "oar_lambda": 1e-4},
        "s": 1.0,
        "t": 1.0,
    }
    layer_options = get_default_layer_options_from_options(options)
    assert layer_options["QRNN_0"]["oar"]["omega"] == 6
    assert layer_options["QRNN_0"]["oar"]["oar_lambda"] == 1e-4
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_phase1_naming.py -v`
Expected: FAIL with missing keys or old names still used.

**Step 3: Write minimal implementation**

```python
# Update all options handling:
# - Replace options["oar"]["precision"] with options["oar"]["omega"]
# - Replace options["oar"]["lm"] with options["oar"]["oar_lambda"]
# - Replace local variables oar_bits -> omega and oar_lm -> oar_lambda
# - Update layer option dicts to use "omega" and "oar_lambda"

# Update import/usage sites:
# - main.py: use sign_ste_tanh and mod_sign
# - mnist_rnn_model.py: use TrackedActivation, Downsampling, OARModel
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_phase1_naming.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_phase1_naming.py utils/model_utils.py main.py mnist_rnn_model.py
git commit -m "refactor: update option keys and call sites"
```

---

### Task 4: Clean up docs and verify no old names remain

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update docs to use new names**

```text
# Replace mentions of TimeReduction with Downsampling (if present).
# Ensure references match new activation and model class names.
```

**Step 2: Verify old names are gone**

Run:
- `rg "oar_hat_fn|oar_hat_metric_fn|mod_sign_with_tanh_deriv|sign_with_tanh_deriv|GeneralActivation|TimeReduction|ModelWithGradInfo|oar_bits|oar_lm|\\blm\\b" -g"*.py"`
Expected: No matches in code.

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: align naming with phase 1 refactor"
```
