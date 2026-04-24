# Self-Pruning Neural Network
A feed-forward network that learns to prune its own weights during training using learnable gates — no post-training pruning needed.
Built for CIFAR-10 image classification as part of the Tredence AI Engineering Internship case study.

## How It Works
Each weight has a gate value between 0 and 1 (via sigmoid). Adding an L1 
penalty on these gates to the loss pushes them toward zero — the optimizer 
ends up killing weights that aren't useful enough to justify keeping.

Total Loss = CrossEntropy + λ × Σ sigmoid(gate_scores)

## Project Structure
```
Self_pruning_nn/
├── model.py           # PrunableLinear layer + network definition
├── train.py           # training loop with custom sparsity loss
├── visualize.py       # gate distribution, accuracy, sparsity, tradeoff plots
├── run_experiment.py  # main entry point — runs all 3 λ values
├── report.md          # results, analysis and observations
├── requirements.txt   # dependencies
└── plots/
    ├── gate_distribution.png
    ├── accuracy.png
    ├── sparsity.png
    └── tradeoff.png
```


## Setup

```bash
pip install -r requirements.txt
python run_experiment.py
```
CIFAR-10 downloads automatically on first run.
## Results

| Lambda | Test Accuracy | Sparsity (%) |
|--------|--------------|--------------|
| 1e-5   | 62.04%       | 37.5%        |
| 1e-4   | 61.72%       | 82.8%        |
| 1e-3   | 62.11%       | 99.1%        |

At λ=1e-3, 99.1% of weights were pruned with no accuracy loss — confirming most network capacity was redundant.
