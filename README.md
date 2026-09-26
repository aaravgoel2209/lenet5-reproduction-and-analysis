# LeNet-5 Reproduction and Analysis

This project reproduces and investigates LeNet-5, based on the 1998 paper
*Gradient-Based Learning Applied to Document Recognition*.

The current implementation is a **modernized LeNet-style baseline**, not yet a
faithful reproduction of the original architecture. It classifies 28 × 28
grayscale MNIST images using ReLU activations and max pooling. Future work will
compare original and modern design choices. The original model and experiments
are placeholders only.

## Setup

Use Python 3.14 or newer, as specified in the existing project configuration.
From the project directory, install the locked environment with:

```bash
uv sync
```

Alternatively, install the baseline's required packages into your environment:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` contains PyTorch and torchvision, using the minimum versions
already declared in `pyproject.toml`. The existing uv configuration also includes
NumPy and Matplotlib; neither is required by the baseline scripts directly.

## Train and evaluate

```bash
python train.py
python evaluate.py
```

With uv, use `uv run python train.py` and `uv run python evaluate.py`.
The scripts use CUDA when available and otherwise CPU. Run these commands from
the project directory because evaluation uses relative file paths. Existing MNIST data in `train/` and `test/`
is reused; missing data is downloaded automatically and requires internet access.

Training retains batch size 256, SGD with learning rate 0.1, cross-entropy loss,
and 100 epochs. Training batches are shuffled each epoch. The previous stopping
rule based on nearly identical consecutive accuracies has been removed, so a
short plateau does not prematurely end training. Edit `all_epoch` in `train.py`
to change the run length. No random seed is set, so results can vary.

Each epoch prints and records the sample-weighted mean training loss and test
accuracy (a fraction from 0 to 1). Evaluation disables gradient tracking and
switches the model to evaluation mode.

| Output | Contents |
| --- | --- |
| `results/metrics/training_metrics.csv` | `epoch`, `train_loss`, `test_accuracy` for every epoch |
| `results/checkpoints/best_model.pt` | Model `state_dict()` saved only when accuracy strictly improves |
| `results/metrics/evaluation.json` | Test accuracy from evaluating the saved best checkpoint |

Each training run replaces the previous training CSV and best checkpoint;
copy outputs elsewhere before starting a run you want to keep separately.
Evaluation replaces `evaluation.json`. Checkpoints are ignored by Git.
Existing legacy `models/*.pkl` files are left untouched, but the new evaluator
loads only the new state-dictionary checkpoint. The old `model.Model` import
path has been removed; legacy full-model pickles require the old implementation.

**Methodology:** this baseline checks the MNIST test set every epoch and uses
its accuracy to choose the checkpoint. Consequently, this test accuracy is used
for model selection and is not an independent final estimate. Running
`evaluate.py` on the same set does not change that. Before research experiments,
create a training/validation split, select models using validation accuracy, and
reserve the test set for final evaluation.

## Architecture

The original baseline's layers are preserved except for removal of the final
ReLU. The last layer returns raw logits for `CrossEntropyLoss`.

| Layer | Output shape (excluding batch) |
| --- | --- |
| Input | `1 × 28 × 28` |
| `Conv2d(1, 6, 5)` + ReLU | `6 × 24 × 24` |
| `MaxPool2d(2)` | `6 × 12 × 12` |
| `Conv2d(6, 16, 5)` + ReLU | `16 × 8 × 8` |
| `MaxPool2d(2)` | `16 × 4 × 4` |
| Flatten | `256` |
| `Linear(256, 120)` + ReLU | `120` |
| `Linear(120, 84)` + ReLU | `84` |
| `Linear(84, 10)` | `10` raw logits |

The predicted digit is the index of the largest logit.

## Project structure

```text
models/
    __init__.py
    modern_lenet.py          ModernLeNet baseline
    original_lenet.py        Placeholder
train.py                    Training and best-checkpoint selection
evaluate.py                 Shared evaluation function and standalone evaluator
experiments/
    __init__.py
    activation_comparison.py  Placeholder
    pooling_comparison.py     Placeholder
    robustness.py             Placeholder
results/
    metrics/
    figures/
    checkpoints/
paper/
    notes.md
requirements.txt
pyproject.toml
uv.lock
train/MNIST/                Existing MNIST data
test/MNIST/                 Existing MNIST data
```

## Acknowledgments

The initial implementation was based on
[ChawDoe's LeNet-5 MNIST PyTorch project](https://github.com/ChawDoe/LeNet-5-MNIST-PyTorch).
See [LICENSE](LICENSE) for this repository's license.
