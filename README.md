# LeNet-5 Reproduction and Analysis

A compact PyTorch implementation of a LeNet-style convolutional neural network for MNIST handwritten digit classification. The model predicts digits from **0 to 9** using grayscale **28 × 28** images.

This implementation uses ReLU activations and max pooling. It is a simplified LeNet-style model rather than an exact reproduction of the original architecture.

## Setup

Use **Python 3.14 or newer** and **uv**. Run these commands to clone the project and install the locked dependencies:

```bash
git clone https://github.com/aaravgoel2209/lenet5-reproduction-and-analysis.git
cd lenet5-reproduction-and-analysis
uv sync
```

Dependencies are declared in `pyproject.toml`: PyTorch, torchvision, NumPy, and Matplotlib. `uv.lock` records the resolved versions.

## Train the model

Run from the project root so the relative dataset and checkpoint paths resolve correctly:

```bash
uv run python train.py
```

The script automatically uses CUDA when available and otherwise runs on the CPU. It trains on 60,000 MNIST images and evaluates on 10,000 test images after each epoch.

| Setting | Value |
| --- | --- |
| Batch size | 256 |
| Optimizer | Stochastic gradient descent (SGD) |
| Learning rate | 0.1 |
| Loss | Cross-entropy |
| Maximum epochs | 100 |
| Early stopping | Absolute accuracy change between consecutive epochs is less than `0.00001` |

The current early-stopping rule compares test accuracy, including both increases and decreases. For experiments requiring an independent final test result, use a separate validation split for stopping decisions.

## Output and checkpoints

After each epoch, the script prints test accuracy as a fraction, rounded to three decimal places:

```text
accuracy: 0.976
```

This means approximately **97.6% accuracy**. Results can vary between runs; the script does not set a random seed.

The entire model is saved after each epoch to `models/mnist_<accuracy>.pkl`, for example `models/mnist_0.976.pkl`. Epochs with the same rounded accuracy overwrite the same file. Checkpoints remain local and are ignored by Git.

## Model architecture

Shapes below omit the batch dimension. Each convolution and fully connected layer is followed by ReLU, including the final output layer.

| Layer | Output shape |
| --- | --- |
| Input | `1 × 28 × 28` |
| Convolution: 6 filters, 5 × 5 | `6 × 24 × 24` |
| Max pooling: 2 × 2 | `6 × 12 × 12` |
| Convolution: 16 filters, 5 × 5 | `16 × 8 × 8` |
| Max pooling: 2 × 2 | `16 × 4 × 4` |
| Flatten | `256` |
| Fully connected | `120` |
| Fully connected | `84` |
| Fully connected | `10` |

The predicted digit is the index of the highest output score.

## Project structure

```text
model.py          Neural network definition
train.py          Training, evaluation, and checkpoint saving
pyproject.toml    Python requirement and dependencies
uv.lock           Locked dependency versions
train/MNIST/raw/  Included MNIST data used by the training loader
test/MNIST/raw/   Included MNIST data used by the test loader
models/           Generated checkpoints (ignored by Git)
```

Both dataset folders are included in the repository. The loaders also use `download=True` to download missing data when needed; this requires internet access. Each folder contains both MNIST splits, and the loader's `train` argument selects the appropriate split.

Virtual environments, Python caches, local editor settings, checkpoints, and the generated `predictions.png` plot are excluded through `.gitignore`.

## Acknowledgments

The initial implementation and README were based on [ChawDoe's LeNet-5 MNIST PyTorch project](https://github.com/ChawDoe/LeNet-5-MNIST-PyTorch). See [LICENSE](LICENSE) for this repository's license.
