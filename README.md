# one-step-gen-empirical

Empirical Tests comparing the accuracy of reconstruction between typical flow matching models and one-step generation (specifically Mean Flow, Improved Mean Flow, and Drifting Models). Everything is designed to run comfortably under a 16GB memory budget (small networks, small/subsetted datasets, CPU-friendly).

## Datasets

The repository includes both synthetic 2D datasets and small real image datasets to evaluate the models. Synthetic datasets are particularly useful for diagnosing exploration and mode collapse.

| Dataset | Type | Description & Purpose |
|---------|------|-----------------------|
| `moons` | 2D Synthetic | Two disjoint, continuous half-moon shapes. Good for basic sanity checking. |
| `checkerboard` | 2D Synthetic | Grid of disjoint squares. Tests a model's ability to maintain sharp boundaries and cover multiple distinct modes. |
| `8gaussians` | 2D Synthetic | Eight Gaussian modes arranged in a circle. Evaluates basic multi-modal coverage. |
| `25gaussians` | 2D Synthetic | Standard 5x5 grid of Gaussians. The de facto standard for testing mode dropping and global exploration in one-step or distilled models. |
| `unequal_gmm` | 2D Synthetic | A 3-component Gaussian Mixture Model with unbalanced probabilities (70%, 20%, 10%) and variances. Ideal for testing if minority modes are ignored. |
| `swissroll` | 2D Synthetic | A continuous, spiraling 2D manifold. |
| `mnist` | Images | Standard handwritten digits (subsetted, in-memory) for evaluating pixel-space generation. |
| `fashionmnist` | Images | Clothing items, slightly more complex than MNIST, used for pixel-space evaluation. |

## Evaluation

We evaluate each one-step method (`mean_flow`, `improved_mean_flow`, `drifting_model`) at 1, 2, and 4 function evaluations (NFE) to observe how they behave as the strict one-step budget is relaxed. The `flow_matching` baseline is evaluated at 1 step (naive single-Euler-step) and at a full step count (default 50).

The following metrics are tracked:

- **Synthetic 2D Data**: 
  - **RBF MMD** (Maximum Mean Discrepancy)
  - **Sliced Wasserstein Distance**
- **Images**: 
  - **Random-feature Frechet Distance** (a proxy for FID that doesn't require downloading an internet-dependent pretrained network)
  - **Pixel-space MMD**
- **General Metrics**: 
  - **Wall-clock sampling time**
  - **NFE** (Number of Function Evaluations)

## Run

The main entry point is `main.py`. The script will automatically save checkpoints and logs to the `weights/` and `logs/` directories.

### Flags
- `--dataset`: The dataset to use (e.g., `moons`, `25gaussians`, `mnist`).
- `--method`: The method(s) to train and evaluate. Choices: `flow_matching`, `mean_flow`, `improved_mean_flow`, `drifting_model`, or `all`. You can also provide a comma-separated list (e.g., `mean_flow,drifting_model`).
- `--smoke_test`: Run a fast sanity check with tiny epochs and sizes.
- `--force`: Force retraining of a method even if a checkpoint already exists in `weights/`. By default, existing checkpoints are just loaded and re-evaluated.
- `--sample_steps`: The number of inference steps for the multi-step `flow_matching` baseline (default: 50).

### Examples

Sanity-check the whole pipeline in seconds:
```bash
python main.py --dataset moons --method all --smoke_test
```

Run all four methods on a synthetic dataset:
```bash
python main.py --dataset 25gaussians --method all
```

Only run specific methods:
```bash
python main.py --dataset checkerboard --method mean_flow,drifting_model
```

Force retraining of the Flow Matching baseline:
```bash
python main.py --dataset unequal_gmm --method flow_matching --force
```

Run on a real image dataset (downloads via torchvision on first run):
```bash
python main.py --dataset mnist --method all
```
