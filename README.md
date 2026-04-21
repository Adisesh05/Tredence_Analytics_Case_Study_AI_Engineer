# Self-Pruning Neural Network

A feed-forward neural network that learns to prune its own weights
during training using learnable gate parameters and L1 sparsity
regularization. This approach eliminates the need for any post-training
pruning step.

Developed as part of the Tredence Analytics AI Engineer Case Study.

------------------------------------------------------------------------

## Project Structure

    .
    ├── self_pruning_nn.py   # Main implementation
    ├── requirements.txt     # Dependencies
    ├── README.md            # Project documentation
    └── outputs/             # Generated results
        ├── gate_distributions.png
        ├── training_curves.png
        └── report.md

------------------------------------------------------------------------

## Quick Start

``` bash
pip install -r requirements.txt
python self_pruning_nn.py
```

All outputs (plots and report) will be saved in:

    ./outputs/

------------------------------------------------------------------------

## Core Idea

Each weight in the network is associated with a learnable gate:

    g_ij = sigmoid(s_ij)

The effective weight becomes:

    w_ij * g_ij

-   Gates are trained jointly with weights\
-   L1 regularization pushes unnecessary gates toward zero\
-   This results in automatic pruning during training

------------------------------------------------------------------------

## Loss Function

    L_total = CrossEntropy + λ × Σ g_ij

-   CrossEntropy ensures classification performance\
-   L1 on gates enforces sparsity

------------------------------------------------------------------------

## Experimental Setup

-   Dataset: CIFAR-10\
-   Architecture: 3072 → 1024 → 512 → 256 → 128 → 10\
-   Model Type: Fully connected (feed-forward)\
-   Optimizer: Adam (separate parameter groups for gates and weights)\
-   Scheduler: Cosine Annealing\
-   Epochs: 30\
-   Lambda values: {1e-4, 3e-4, 6e-4}\
-   Sparsity threshold: 0.05\
-   Device: CUDA

------------------------------------------------------------------------

## Results

| Lambda (λ) | Test Accuracy | Sparsity (%) |
|-----------|--------------|-------------|
| 0.0001    | 57.19%       | 92.9%       |
| 0.0003    | 57.58%       | 98.8%       |
| 0.0006    | 57.34%       | 99.8%       |

------------------------------------------------------------------------

## Training Behavior

Early epochs: - Network behaves like a dense model\
- Sparsity remains near 0%

Mid training (\~epoch 15): - Rapid increase in sparsity\
- Gates begin collapsing toward zero

Later epochs: - Sparsity stabilizes at very high levels\
- Accuracy remains stable

------------------------------------------------------------------------

## Key Observations

-   Increasing λ significantly increases sparsity\
-   Up to 99.8% sparsity achieved\
-   Accuracy remains stable (\~57%) despite heavy pruning\
-   Indicates high redundancy in dense neural networks\
-   Pruning emerges naturally during training, not as a separate step

------------------------------------------------------------------------

## Threshold Note

-   Sparsity is measured using a threshold of 0.05\
-   Lower thresholds (e.g., 1e-2) underestimate pruning\
-   0.05 better captures effectively inactive connections

------------------------------------------------------------------------

## Outputs

Generated files are stored in `outputs/`:

-   `gate_distributions.png`\
-   `training_curves.png`\
-   `report.md`

------------------------------------------------------------------------

## Conclusion

The model demonstrates dynamic self-pruning, achieving extremely high
sparsity with minimal loss in accuracy. This highlights the redundancy
present in dense neural networks and the effectiveness of L1-based
gating mechanisms.

------------------------------------------------------------------------

## Reproducibility

-   Seed: 42\
-   Deterministic CUDA settings enabled\
-   No post-processing or manual pruning applied\
-   All sparsity emerges directly from training dynamics
