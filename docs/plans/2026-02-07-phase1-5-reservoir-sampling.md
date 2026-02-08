# Phase 1.5 Reservoir Sampling Implementation Plan

> **For Agent:** REQUIRED: Use the executing-plans skill to implement this plan task-by-task.

**Goal:** Replace biased EMA preactivation tracking with per-layer reservoir sampling and epoch-end histogram logging.

**Architecture:** Each OAR layer maintains its own fixed-size reservoir (`preact_reservoir`) and counter (`reservoir_count`), updated in `call()` using Vitter's Algorithm R. A lightweight callback logs the per-layer reservoir histogram at epoch end. Reservoir size is configurable per-layer via `reservoir_size` init kwarg (default 100,000), propagated from options. Logging cadence is once per epoch.

**Tech Stack:** Python 3.10, TensorFlow 2.10.1, QKeras 0.9.0, pytest

---

### Task 1: Add reservoir sampling helper test

**Files:**
- Create: `tests/test_reservoir_sampling.py`

**Step 1: Write the failing test**

```python
import tensorflow as tf

from utils.model_utils import _reservoir_update


def test_reservoir_update_fills_initial_slots():
    reservoir = tf.zeros([10], dtype=tf.float32)
    count = tf.constant(0, dtype=tf.int64)
    new_vals = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)

    updated, updated_count = _reservoir_update(
        reservoir=reservoir,
        count=count,
        new_values=new_vals,
        reservoir_size=10,
        seed=123,
    )

    tf.debugging.assert_equal(updated_count, tf.constant(4, dtype=tf.int64))
    tf.debugging.assert_equal(updated[:4], new_vals)


def test_reservoir_update_replaces_after_full():
    reservoir = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0], dtype=tf.float32)
    count = tf.constant(5, dtype=tf.int64)
    new_vals = tf.constant([99.0, 99.0, 99.0], dtype=tf.float32)

    updated, updated_count = _reservoir_update(
        reservoir=reservoir,
        count=count,
        new_values=new_vals,
        reservoir_size=5,
        seed=42,
    )

    tf.debugging.assert_equal(updated_count, tf.constant(8, dtype=tf.int64))
    # Can't assert exact values due to randomness, but shape should be preserved
    assert updated.shape.as_list() == [5]
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_reservoir_sampling.py -v`
Expected: FAIL with "cannot import name '_reservoir_update'".

**Step 3: Commit**

```bash
git add tests/test_reservoir_sampling.py
git commit -m "test: add reservoir sampling helper expectations"
```

---

### Task 2: Implement reservoir sampling helper

**Files:**
- Modify: `utils/model_utils.py`

**Step 1: The tests from Task 1 should still fail**

**Step 2: Write minimal implementation**

Add this function near the top of `utils/model_utils.py` (after imports):

```python
def _reservoir_update(reservoir, count, new_values, reservoir_size, seed):
    """
    Vitter's Algorithm R for reservoir sampling.
    
    Args:
        reservoir: Current reservoir tensor [reservoir_size]
        count: Total samples seen so far (int64 scalar)
        new_values: New values to potentially add (any shape, will be flattened)
        reservoir_size: Size of reservoir (int)
        seed: Random seed for reproducibility
    
    Returns:
        (updated_reservoir, updated_count)
    """
    flat = tf.reshape(new_values, [-1])
    n = tf.size(flat, out_type=tf.int64)
    reservoir_size_i64 = tf.cast(reservoir_size, tf.int64)

    # Phase 1: Fill initial slots if reservoir not full
    capacity_left = reservoir_size_i64 - count
    fill_n = tf.minimum(n, tf.maximum(capacity_left, tf.constant(0, dtype=tf.int64)))
    
    indices = tf.reshape(tf.range(count, count + fill_n, dtype=tf.int64), [-1, 1])
    reservoir = tf.tensor_scatter_nd_update(reservoir, indices, flat[:fill_n])
    new_count = count + fill_n

    # Phase 2: Reservoir sampling for values beyond capacity
    remaining = flat[fill_n:]
    remaining_n = tf.size(remaining, out_type=tf.int64)

    def do_sampling():
        nonlocal reservoir, new_count
        # For element i (0-indexed in remaining), total index is new_count + i
        # Accept with probability reservoir_size / (new_count + i + 1)
        # If accepted, replace random slot in reservoir
        i_range = tf.range(remaining_n, dtype=tf.int64)
        total_idx = new_count + i_range + 1  # 1-indexed for probability calc
        
        # Draw random values to decide acceptance and slot
        rng = tf.random.stateless_uniform(
            shape=[remaining_n, 2],
            seed=[seed, tf.cast(new_count, tf.int32)],
            dtype=tf.float64,
        )
        accept_threshold = tf.cast(reservoir_size_i64, tf.float64) / tf.cast(total_idx, tf.float64)
        accept_mask = rng[:, 0] < accept_threshold
        
        # For accepted values, pick slot uniformly
        slots = tf.cast(rng[:, 1] * tf.cast(reservoir_size_i64, tf.float64), tf.int64)
        slots = tf.minimum(slots, reservoir_size_i64 - 1)  # Clamp to valid range
        
        accepted_slots = tf.boolean_mask(slots, accept_mask)
        accepted_values = tf.boolean_mask(remaining, accept_mask)
        
        indices = tf.reshape(accepted_slots, [-1, 1])
        return tf.tensor_scatter_nd_update(reservoir, indices, accepted_values), new_count + remaining_n

    reservoir, new_count = tf.cond(
        remaining_n > 0,
        do_sampling,
        lambda: (reservoir, new_count),
    )

    return reservoir, new_count
```

**Step 3: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_reservoir_sampling.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add utils/model_utils.py
git commit -m "feat: add reservoir sampling helper"
```

---

### Task 3: Add layer reservoir weight tests

**Files:**
- Modify: `tests/test_reservoir_sampling.py`

**Step 1: Write the failing test**

Append to `tests/test_reservoir_sampling.py`:

```python
from utils.model_utils import QSimpleRNNCellWithOAR, QDenseWithOAR


def test_qsimple_rnn_cell_has_reservoir_weights_default():
    cell = QSimpleRNNCellWithOAR(units=4, batch_size=2, name="QRNN_0")
    cell.build(tf.TensorShape([2, 8]))
    assert cell.preact_reservoir.shape.as_list() == [100000]
    assert cell.reservoir_count.dtype == tf.int64


def test_qsimple_rnn_cell_has_reservoir_weights_custom():
    cell = QSimpleRNNCellWithOAR(units=4, batch_size=2, reservoir_size=500, name="QRNN_0")
    cell.build(tf.TensorShape([2, 8]))
    assert cell.preact_reservoir.shape.as_list() == [500]


def test_qdense_has_reservoir_weights_default():
    layer = QDenseWithOAR(units=3, batch_size=2, name="DENSE_0")
    layer.build(tf.TensorShape([2, 8]))
    assert layer.preact_reservoir.shape.as_list() == [100000]
    assert layer.reservoir_count.dtype == tf.int64


def test_qdense_has_reservoir_weights_custom():
    layer = QDenseWithOAR(units=3, batch_size=2, reservoir_size=1000, name="DENSE_0")
    layer.build(tf.TensorShape([2, 8]))
    assert layer.preact_reservoir.shape.as_list() == [1000]
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_reservoir_sampling.py::test_qsimple_rnn_cell_has_reservoir_weights_default -v`
Expected: FAIL with "unexpected keyword argument 'reservoir_size'" or missing attribute.

**Step 3: Commit**

```bash
git add tests/test_reservoir_sampling.py
git commit -m "test: add layer reservoir weight expectations"
```

---

### Task 4: Replace EMA with reservoir in QSimpleRNNCellWithOAR

**Files:**
- Modify: `utils/model_utils.py:208-350` (QSimpleRNNCellWithOAR class)

**Step 1: The tests from Task 3 should still fail**

**Step 2: Write minimal implementation**

In `QSimpleRNNCellWithOAR.__init__()`, add parameter and store it:

```python
def __init__(
    self,
    units,
    batch_size,
    activation=tf.keras.activations.tanh,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    recurrent_initializer="orthogonal",
    bias_initializer="zeros",
    kernel_quantizer=None,
    recurrent_quantizer=None,
    bias_quantizer=None,
    oar_lambda=0,
    omega=32,
    s=1.0,
    reservoir_size=100000,
    **kwargs,
):
    # ... existing code ...
    self.reservoir_size = reservoir_size
```

In `QSimpleRNNCellWithOAR.build()`, replace the `preacts` EMA weight with reservoir weights:

```python
# Remove this:
# self.preacts = self.add_weight(
#     name="preacts",
#     shape=[self.batch_size, self.units],
#     ...
# )

# Add these:
self.preact_reservoir = self.add_weight(
    name="preact_reservoir",
    shape=[self.reservoir_size],
    dtype=self.kernel.dtype,
    initializer="zeros",
    trainable=False,
)
self.reservoir_count = self.add_weight(
    name="reservoir_count",
    shape=[],
    dtype=tf.int64,
    initializer="zeros",
    trainable=False,
)
```

In `QSimpleRNNCellWithOAR.call()`, replace the EMA update with reservoir update:

```python
# Remove this:
# self.preacts.assign(0.90 * self.preacts + 0.10 * (h + h_2))

# Add this (where h + h_2 is the pre-activation):
preact = h + h_2
updated_reservoir, updated_count = _reservoir_update(
    reservoir=self.preact_reservoir,
    count=tf.cast(self.reservoir_count, tf.int64),
    new_values=preact,
    reservoir_size=self.reservoir_size,
    seed=1997,
)
self.preact_reservoir.assign(updated_reservoir)
self.reservoir_count.assign(updated_count)
```

**Step 3: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_reservoir_sampling.py::test_qsimple_rnn_cell_has_reservoir_weights_default tests/test_reservoir_sampling.py::test_qsimple_rnn_cell_has_reservoir_weights_custom -v`
Expected: PASS

**Step 4: Commit**

```bash
git add utils/model_utils.py
git commit -m "refactor: replace EMA with reservoir in QSimpleRNNCellWithOAR"
```

---

### Task 5: Replace EMA with reservoir in QDenseWithOAR

**Files:**
- Modify: `utils/model_utils.py:365-485` (QDenseWithOAR class)

**Step 1: The QDense tests from Task 3 should still fail**

**Step 2: Write minimal implementation**

In `QDenseWithOAR.__init__()`, add parameter and store it:

```python
def __init__(
    self,
    units,
    batch_size,
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_quantizer=None,
    bias_quantizer=None,
    oar_lambda=0,
    omega=32,
    s=1.0,
    reservoir_size=100000,
    **kwargs,
):
    # ... existing code ...
    self.reservoir_size = reservoir_size
```

In `QDenseWithOAR.build()`, replace the `preacts` EMA weight with reservoir weights:

```python
# Remove this:
# self.preacts = self.add_weight(
#     name="preacts",
#     shape=[self.batch_size, self.units],
#     ...
# )

# Add these:
self.preact_reservoir = self.add_weight(
    name="preact_reservoir",
    shape=[self.reservoir_size],
    dtype=self.kernel.dtype,
    initializer="zeros",
    trainable=False,
)
self.reservoir_count = self.add_weight(
    name="reservoir_count",
    shape=[],
    dtype=tf.int64,
    initializer="zeros",
    trainable=False,
)
```

In `QDenseWithOAR.call()`, replace the EMA update with reservoir update:

```python
# Remove this:
# self.preacts.assign(0.90 * self.preacts + 0.10 * h)

# Add this (where h is the pre-activation):
updated_reservoir, updated_count = _reservoir_update(
    reservoir=self.preact_reservoir,
    count=tf.cast(self.reservoir_count, tf.int64),
    new_values=h,
    reservoir_size=self.reservoir_size,
    seed=1997,
)
self.preact_reservoir.assign(updated_reservoir)
self.reservoir_count.assign(updated_count)
```

**Step 3: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_reservoir_sampling.py::test_qdense_has_reservoir_weights_default tests/test_reservoir_sampling.py::test_qdense_has_reservoir_weights_custom -v`
Expected: PASS

**Step 4: Commit**

```bash
git add utils/model_utils.py
git commit -m "refactor: replace EMA with reservoir in QDenseWithOAR"
```

---

### Task 6: Propagate reservoir_size through QRNNWithOAR

**Files:**
- Modify: `utils/model_utils.py:25-80` (QRNNWithOAR class)

**Step 1: Write the failing test**

Append to `tests/test_reservoir_sampling.py`:

```python
from utils.model_utils import QRNNWithOAR


def test_qrnn_with_oar_propagates_reservoir_size():
    layer = QRNNWithOAR(units=4, batch_size=2, reservoir_size=777, name="QRNN_0")
    layer.build(tf.TensorShape([2, 10, 8]))
    assert layer.cell.preact_reservoir.shape.as_list() == [777]
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_reservoir_sampling.py::test_qrnn_with_oar_propagates_reservoir_size -v`
Expected: FAIL with "unexpected keyword argument 'reservoir_size'".

**Step 3: Write minimal implementation**

In `QRNNWithOAR.__init__()`:

```python
def __init__(
    self,
    units,
    batch_size,
    activation=tf.keras.activations.tanh,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    recurrent_initializer="orthogonal",
    bias_initializer="zeros",
    kernel_quantizer=None,
    recurrent_quantizer=None,
    bias_quantizer=None,
    oar_lambda=0,
    omega=32,
    s=1.0,
    reservoir_size=100000,
    **kwargs,
):
    self.reservoir_size = reservoir_size
    # ... and pass it to cell creation:
    cell = QSimpleRNNCellWithOAR(
        units=units,
        batch_size=batch_size,
        # ... other params ...
        reservoir_size=reservoir_size,
        **kwargs,
    )
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_reservoir_sampling.py::test_qrnn_with_oar_propagates_reservoir_size -v`
Expected: PASS

**Step 5: Commit**

```bash
git add utils/model_utils.py tests/test_reservoir_sampling.py
git commit -m "feat: propagate reservoir_size through QRNNWithOAR"
```

---

### Task 7: Add ReservoirHistogramCallback test

**Files:**
- Modify: `tests/test_reservoir_sampling.py`

**Step 1: Write the failing test**

Append to `tests/test_reservoir_sampling.py`:

```python
import tempfile
from utils.model_utils import ReservoirHistogramCallback


def test_reservoir_histogram_callback_constructs():
    with tempfile.TemporaryDirectory() as tmpdir:
        cb = ReservoirHistogramCallback(log_dir=tmpdir)
        assert cb is not None
        assert cb.log_dir == tmpdir
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_reservoir_sampling.py::test_reservoir_histogram_callback_constructs -v`
Expected: FAIL with "cannot import name 'ReservoirHistogramCallback'".

**Step 3: Commit**

```bash
git add tests/test_reservoir_sampling.py
git commit -m "test: add ReservoirHistogramCallback expectation"
```

---

### Task 8: Implement ReservoirHistogramCallback

**Files:**
- Modify: `utils/model_utils.py`

**Step 1: The test from Task 7 should still fail**

**Step 2: Write minimal implementation**

Add this class to `utils/model_utils.py`:

```python
class ReservoirHistogramCallback(tf.keras.callbacks.Callback):
    """Logs preact_reservoir histograms to TensorBoard at epoch end."""

    def __init__(self, log_dir, reservoir_weight_name="preact_reservoir"):
        super().__init__()
        self.log_dir = log_dir
        self.reservoir_weight_name = reservoir_weight_name
        self.writer = None

    def set_model(self, model):
        super().set_model(model)
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        if self.writer is None:
            return
        with self.writer.as_default():
            for weight in self.model.weights:
                if self.reservoir_weight_name in weight.name:
                    tf.summary.histogram(weight.name, weight, step=epoch)
            self.writer.flush()
```

**Step 3: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_reservoir_sampling.py::test_reservoir_histogram_callback_constructs -v`
Expected: PASS

**Step 4: Commit**

```bash
git add utils/model_utils.py
git commit -m "feat: add ReservoirHistogramCallback"
```

---

### Task 9: Wire ReservoirHistogramCallback into main.py

**Files:**
- Modify: `main.py`

**Step 1: No new test needed (integration)**

**Step 2: Write minimal implementation**

Import at top of `main.py`:

```python
from utils.model_utils import ReservoirHistogramCallback
```

In the training setup (near the existing `tb_callback`), add:

```python
reservoir_cb = ReservoirHistogramCallback(log_dir=(RUN_DIR + TB_LOGS_DIR))
```

Add `reservoir_cb` to the callbacks list passed to `model.fit()`.

**Step 3: Verify main.py runs without error**

Run: `PYTHONPATH=. python -c "import main; print('import ok')"`
Expected: "import ok" (no syntax errors).

**Step 4: Commit**

```bash
git add main.py
git commit -m "feat: wire ReservoirHistogramCallback into training"
```

---

### Task 10: Add reset_stat_weights helper test

**Files:**
- Modify: `tests/test_reservoir_sampling.py`

**Step 1: Write the failing test**

Append to `tests/test_reservoir_sampling.py`:

```python
from utils.model_utils import reset_stat_weights


def test_reset_stat_weights_zeros_reservoir():
    class DummyModel:
        def __init__(self):
            self.weights = [
                tf.Variable([1.0, 2.0, 3.0], name="layer/preact_reservoir:0"),
                tf.Variable([5], dtype=tf.int64, name="layer/reservoir_count:0"),
                tf.Variable([9.0], name="layer/kernel:0"),  # should NOT be zeroed
            ]

        def get_weights(self):
            return [w.numpy() for w in self.weights]

        def set_weights(self, new_weights):
            for w, nw in zip(self.weights, new_weights):
                w.assign(nw)

    model = DummyModel()
    reset_stat_weights(model)
    tf.debugging.assert_equal(model.weights[0], tf.zeros([3]))
    tf.debugging.assert_equal(model.weights[1], tf.zeros([], dtype=tf.int64))
    tf.debugging.assert_equal(model.weights[2], tf.constant([9.0]))  # unchanged
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_reservoir_sampling.py::test_reset_stat_weights_zeros_reservoir -v`
Expected: FAIL with "cannot import name 'reset_stat_weights'".

**Step 3: Commit**

```bash
git add tests/test_reservoir_sampling.py
git commit -m "test: add reset_stat_weights expectation"
```

---

### Task 11: Implement reset_stat_weights helper

**Files:**
- Modify: `utils/model_utils.py`

**Step 1: The test from Task 10 should still fail**

**Step 2: Write minimal implementation**

Add this function to `utils/model_utils.py`:

```python
def reset_stat_weights(model):
    """Zero out stat-tracking weights (wx, preact_reservoir, reservoir_count)."""
    weights = model.get_weights()
    for i in range(len(weights)):
        name = model.weights[i].name
        if any(pattern in name for pattern in ["/w", "/x", "preact_reservoir", "reservoir_count"]):
            weights[i] = 0 * weights[i]
    model.set_weights(weights)
```

**Step 3: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_reservoir_sampling.py::test_reset_stat_weights_zeros_reservoir -v`
Expected: PASS

**Step 4: Commit**

```bash
git add utils/model_utils.py
git commit -m "feat: add reset_stat_weights helper"
```

---

### Task 12: Use reset_stat_weights in main.py and export script

**Files:**
- Modify: `main.py`
- Modify: `export_mnist_weights_h5.py`

**Step 1: No new test needed (integration)**

**Step 2: Write minimal implementation**

In `main.py`, import and replace the inline reset block:

```python
from utils.model_utils import reset_stat_weights

# Replace this block:
# weights = model.get_weights()
# for i in range(len(weights)):
#     if ("/w" in model.weights[i].name or ...):
#         weights[i] = 0 * weights[i]
# model.set_weights(weights)

# With:
reset_stat_weights(model)
```

Do the same in `export_mnist_weights_h5.py`.

**Step 3: Verify scripts run without error**

Run: `PYTHONPATH=. python -c "import main; import export_mnist_weights_h5; print('imports ok')"`
Expected: "imports ok".

**Step 4: Commit**

```bash
git add main.py export_mnist_weights_h5.py
git commit -m "refactor: use reset_stat_weights helper"
```

---

### Task 13: Run all tests and verify

**Files:**
- None (verification only)

**Step 1: Run all reservoir tests**

Run: `PYTHONPATH=. pytest tests/test_reservoir_sampling.py -v`
Expected: All tests PASS.

**Step 2: Run Phase 1 naming tests (regression check)**

Run: `PYTHONPATH=. pytest tests/test_phase1_naming.py -v`
Expected: All tests PASS.

**Step 3: Commit docs update**

Update `CLAUDE.md` to mention reservoir sampling if needed, then:

```bash
git add CLAUDE.md
git commit -m "docs: note reservoir sampling for preact histograms"
```
