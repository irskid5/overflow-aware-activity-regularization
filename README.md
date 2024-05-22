# Overflow-Aware Activity Regularization

This code base implements the overflow-aware activity regularizer (OAR) for moving pre-activations in quantized RNNs to correct regions post-sign function evaluation.

## Preliminaries

The experiments should be run on Ubuntu 22.04.2 LTS,
5.19.0-46-generic. Ensure that python version == 3.10.6 and that, if you are using GPUs, the CUDA version is 11.2.152.

Steps:

1. Install a virtual environment: `python -m venv .venv`
2. Activate virtual environment: `source .venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`

## Evaluation instructions

Running `main.py` evaluates all plaintext experiments in the paper, besides those over encrypted data, and extracts the models and dataset used for evaluation over encrypted data.

```
if __name__ == "__main__":
    train_quantize_extract_MNIST_RNN()
    train_quantize_extract_enlarged_MNIST_RNN()
    evaluation_with_and_without_oar2()
    evaluation_different_oar_regularization_rates()
    extract_ternarized_mnist_test_dataset()
```

Above is the block of code executed in `main.py`

1. `train_quantize_extract_MNIST_RNN()`: This function trains, quantizes, and extracts the MNIST RNN using OAR 6-bit at a rate of 1e-4. The .hdf5 file that contains the quantized parameters is located in the resulting `runs/<year_month>/<datetime>/checkpoints/hdf5` folder. This can be put into the `tfhe_rnns_mnist/models/regular` folder in the supplementary zip file to run the model over encrypted data.

2. `train_quantize_extract_enlarged_MNIST_RNN()`: This function trains, quantizes, and extracts the enlarged MNIST RNN using OAR 6-bit at a rate of 1e-4. The .hdf5 file that contains the quantized parameters is located in the resulting `runs/year_month/datetime/checkpoints/hdf5` folder. This can be put into the `tfhe_rnns_mnist/models/enlarged` folder in the supplementary zip file to run the model over encrypted data.

3. `evaluation_with_and_without_oar2()`: This function evaluates the experiment in section 4.1.1.

4. `evaluation_different_oar_regularization_rates()`: This function evaluates the experiment in section 4.1.2.

5. `extract_ternarized_mnist_test_dataset()`: This function extracts the preprocessed, ternarized MNIST test dataset in [28,28] and [128,128] format, and places the files in the root folder `mnist_preprocessed`. This folder is to be placed in `tfhe_rnns_mnist` to use those datasets for evaluation over encrypted data. They have already been added to the folder.


Steps 1 and 2 record a vast amount of metrics in graph form in TensorBoard. To display them, run 

`tensorboard --logdir runs/<year_month>`

and navigate across the display. Here, you will be able to find accuracy, loss, and OAR metrics. If you navigate to the histograms tab, you will be able to search `preact` and see the pre-activation distributions from step 4 of the 4-step quantization process to recreate the figures in section 4.1.3.

Section 4.1.4 metrics can also be extracted. After each training run in all experiments, there is a run over the test dataset to get test set metrics, which you are able to see in the console output.