# nn-numpy

A two-layer neural network built **from scratch using only NumPy** — no PyTorch, no TensorFlow. Trained on the [Kaggle Digit Recogniser (MNIST)](https://www.kaggle.com/competitions/digit-recognizer) dataset.

The goal is to deeply understand the mathematics of a neural network by implementing every component by hand: weight initialisation, forward propagation, backpropagation, and gradient descent.

---

## Architecture

```
Input (784)  →  Hidden Layer (10, ReLU)  →  Output (10, Softmax)
```

| Component | Detail |
|---|---|
| Input size | 784 (28×28 pixels, normalised to [0, 1]) |
| Hidden units | 10 |
| Output units | 10 (digits 0–9) |
| Activation (hidden) | ReLU |
| Activation (output) | Softmax |
| Loss | Cross-entropy |
| Optimiser | Mini-batch gradient descent |

---

## Files

| File | Purpose |
|---|---|
| `main.py` | Full training + validation pipeline |
| `pytorch_nn.py` | Equivalent model in PyTorch (for comparison) |
| `np_tut.py` | NumPy tutorial scratch file |
| `train.csv` | Training data (Kaggle MNIST format) |
| `test.csv` | Test data |

---

## Usage

### Prerequisites

```bash
pip install numpy pandas matplotlib
```

### Run training

```bash
python main.py
```

Every 20 iterations the script prints loss and training accuracy. After training, validation accuracy is reported on a held-out 1 000-sample dev set.

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate | `0.05` |
| Iterations | `1000` |
| Train / dev split | 41 000 / 1 000 |

---

## What this teaches

- Manual forward propagation through each layer
- Deriving and implementing backprop for ReLU + Softmax + cross-entropy
- Weight update via gradient descent (no autograd)
- The relationship between NumPy matrix shapes and neural network dimensions

---

## Part of the `nn-*` series

| Repo | Description |
|---|---|
| **nn-numpy** | This repo — bare NumPy, MNIST |
| [nn-v1](https://github.com/sobitkarki1/nn-v1) | Structured rewrite with configurable architecture |
| [nn-v2](https://github.com/sobitkarki1/nn-v2) | Added regularisation and new layer types |
| [nn-v4](https://github.com/sobitkarki1/nn-v4) | 1.5B-parameter GPT-style transformer |

---

## License

MIT