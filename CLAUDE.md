# Overflow-Aware Activity Regularization (OAR)

Research codebase implementing OAR for quantized RNNs on MNIST. Produces models for TFHE (homomorphic encryption) inference.

## Quick Start

```bash
# Create venv with Python 3.10 (required for TF 2.10.1 compatibility)
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# For GPU support, use the activation script instead:
source activate_gpu.sh

# Run experiments
python main.py
```

## Environment

- **Python 3.10.x** (required - TF 2.10.1 doesn't support Python 3.11+)
- TensorFlow 2.10.1, QKeras 0.9.0
- GPU: NVIDIA with CUDA support (optional, uses pip-installed CUDA 11 libraries)

## Architecture

```
main.py                 # Entry point, training loops, four-step quantization
mnist_rnn_model.py      # Model architecture (QRNN + Dense layers)
quantization.py         # TernarizationWithThreshold quantizer
utils/model_utils.py    # TrackedActivation, Downsampling, OARModel, OAR1/OAR2
export_mnist.py         # Preprocess MNIST for TFHE inference
export_mnist_weights_h5.py  # Export weights to HDF5
```

**Model Structure:**
- Input → Ternarize (optional) → QRNN_0 → Downsampling → QRNN_1 → Flatten → DENSE_0 → DENSE_OUT

## Key Concepts

### Four-Step Quantization Process
1. **Step 1**: Train with tanh activation (baseline)
2. **Step 2**: Replace tanh with sign (tanh gradient), apply gradient scaling `s`
3. **Step 3**: Ternarize inputs
4. **Step 4**: Enable weight quantization + OAR (learning rate ×0.1)

### Options Dictionary
```python
options = {
    "enlarge": False,      # 28x28 (False) or 128x128 (True)
    "epochs": 100,
    "learning_rate": 1e-4,
    "batch_size": 512,
    "t": 1.5,              # Ternarization threshold multiplier
    "tᵢ": 0.7,             # Input ternarization threshold
    "s": 4.0,              # Gradient scaling factor
    "oar": {
        "oar_lambda": 1e-4,  # OAR regularization rate
        "omega": 6,          # Bit precision (2^omega modulus)
    },
    "quantize": False,     # Enable weight ternarization
}
```

### OAR Regularizers
- `OAR1`: Linear penalty for overflow regions
- `OAR2`: Squared penalty (used in experiments) - penalizes pre-activations outside valid modular range

## Commands

```bash
# Activate environment (with GPU support)
source activate_gpu.sh

# Run all experiments (four-step quantization, ~2200 epochs total)
python main.py

# View TensorBoard metrics
tensorboard --logdir runs/ --port 6006

# Key TensorBoard searches:
# - "preact" in Histograms: pre-activation distributions
# - "OAR2": overflow regularization metrics
```

## Output Structure

```
runs/<YYYYMM>/<YYYYMMDD-HHMMSS>/
├── logs/tensorboard/    # TensorBoard logs
└── checkpoints/
    └── hdf5/
        └── weights.hdf5  # Exported for TFHE inference
```

## Gotchas

- Seed is fixed (`tf.random.set_seed(1997)`) for reproducibility
- `RECORD_CKPTS = True` controls checkpoint saving (edit in main.py)
- Batch size must match between training and model construction (used for stat tracking weights)
- Unicode chars in code: `τ` (tau), `tᵢ` (t_i) - ensure UTF-8 encoding
- QKeras quantizers use STE (straight-through estimator) by default
- **GPU setup**: Use `source activate_gpu.sh` (not just `source .venv/bin/activate`) to set `LD_LIBRARY_PATH` and `XLA_FLAGS` for pip-installed CUDA libraries
