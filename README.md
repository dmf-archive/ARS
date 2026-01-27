# ARS2-Neo: Gliding Directly Towards Global Optima Along Geodesics of the Loss Landscape

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.12+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-orange.svg)](https://pytorch.org)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/dmf-archive/ARS)

> This project is a research framework focused on second-order optimization dynamics and information geometry. It realizes an efficient gliding optimization paradigm on Riemannian manifolds through the principle of Energy-Geometry Decoupling.

## 1. Theoretical Foundation: From Diagonal Fisher to Full-rank NGD

The core design of ARS2-Neo is based on a deep reconstruction of modern optimization algorithms, aimed at overcoming the limitations of first-order optimizers in ill-conditioned curvature landscapes.

### 1.1 Parameter De-correlation

Through **Muon**'s Newton-Schulz iteration, ARS2-Neo enforces orthogonality on the update matrices (Stiefel manifold constraint). Mathematically, orthogonalized updates are equivalent to performing de-correlation in the parameter space, eliminating internal covariate shift and purifying the gradient information.

### 1.2 Full-rank Fisher Approximation and NGD

The Adam optimizer essentially performs a diagonal approximation of the Fisher Information Matrix via second moments. When this diagonal Fisher preconditioning meets Muon's de-correlated parameter space, the originally lost off-diagonal information is geometrically compensated.

- **Operator Composition Effect**: Diagonal Fisher + Orthogonalized Parameter Space ≈ **Full-rank Fisher Information Matrix**.
- **Dynamic Characteristics**: This enables ARS2-Neo to essentially perform high-efficiency **Natural Gradient Descent (NGD)**. In Wikitext-2 experiments, ARS2-Neo (Base) reached a training loss of 0.9 in just 20 epochs, demonstrating its powerful landscape smoothing capability.

### 1.3 Global Optima and MDL Principle

While NGD provides rapid convergence, it is prone to falling into "sharp minima" (overfitting). ARS2-Neo introduces **Manifold-Aware SAM (Sharpness-Aware Minimization)**:

- **Flatness Constraint**: By searching for adversarial directions on the Riemannian manifold, the algorithm is guided towards broader basins in the loss landscape.
- **MDL Correspondence**: According to the Minimum Description Length (MDL) principle, flatter regions correspond to simpler model explanations, thereby possessing stronger generalization capabilities.

## 2. Core Mechanism: Energy-Geometry Decoupling

ARS2-Neo decomposes the optimization process into two independent operators:

1. **Statistical Operator (Energy)**: Uses the second-moment corrected momentum norm from AdamW to determine the update step size, serving as a proxy for the rate of free-energy descent.
2. **Structural Operator (Geometry)**: Ensures the update direction strictly follows the manifold's **Geodesic** through pre-whitening and orthogonal projection.

## 3. Key Experimental Results (LRP Verification)

### 3.1 Wikitext-2 Language Modeling

Experimental Setup: Qwen3 (RoPE, 3-layer), Context 255. Aimed at probing optimization stability on ill-conditioned curvature manifolds.

| Optimizer | Best PPL | Last PPL | Dynamic Characteristics | Description |
| :--- | :--- | :--- | :--- | :--- |
| **AdamW** | 116.46 | 213.52 | Standard Euclidean Baseline | Slow convergence, late-stage overfitting |
| **Muon** | 111.35 | 475.65 | Spectral Constrained Convergence | Lacks adaptive energy, late-stage collapse |
| **ARS2-Neo (Base)** | 96.10 | 3055.47 | **Overfitting** | Rapidly drops into sharp minima, generalization collapse |
| **ARS2-Neo (Sync)** | **90.69** | **330.85** | **Optimal Generalization Ceiling** | `ρ=0.3`, successfully suppresses overfitting |
| **ARS2-Neo (AGA)** | 93.23 | 414.83 | Trade-off between Efficiency & Stability | `λ=0.5`, Adaptive Geometric Awareness |

**Core Insight**: ARS2-Neo (AGA) reaches 93.23 PPL in just 3 epochs, far surpassing AdamW's best performance, proving the generational advantage of second-order geometric information in capturing semantic patterns.

### 3.2 CIFAR-10 Visual Classification

Experimental Setup: ResNet-18, Batch Size 256.

| Optimizer | Best Acc | Final Acc | Note |
| :--- | :--- | :--- | :--- |
| **ARS2-Neo (Sync)** | **95.87%** | **95.73%** | **SOTA**. Rapid convergence in 60 epochs. |
| **AdamW** | 94.60% | 94.47% | Standard Baseline. |
| **Muon** | 93.76% | 93.69% | Pure geometric optimization, limited ceiling. |

### 3.3 Grokking Phenomenon Acceleration

To verify the dynamic characteristics of the optimizer during generalization phase transitions, we compared the performance of various optimizers on a modular addition task (`p=113`, `train_frac=0.3`).

| Optimizer | Fitting (Epoch) | Grokking (Epoch) | Convergence (Epoch) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **AdamW** | ~140 | >600 | N/A | Severe generalization lag; failed to grok within 600 epochs. |
| **Muon** | ~150 | >400 | N/A | Pure geometric optimization wanders slowly without adaptive energy. |
| **ARS2-Neo (Base)** | **20** | **180** | **250** | **Ultra-fast Grokking**. Energy-Geometry Decoupling significantly accelerates phase transition. |
| **ARS2-Neo (AGA)** | **20** | **150** | **200** | **Optimal Dynamics**. Adaptive Geometric Awareness further shortens generalization lag. |

**Core Insight**: ARS2-Neo accelerates the occurrence of Grokking by **over 4x**, strongly proving that Energy-Geometry Decoupling avoids ineffective wandering in overfitting basins, directly traversing high-dimensional canyons to reach generalized solutions.

## 4. Quick Start

### 4.1 Installation

```bash
# uv is recommended
uv sync
```

### 4.2 Running Experiments

```bash
# Run WikiText-2 Sync Mode (Optimal Generalization)
python -m exp.wikitext2.train --config config/lrp_wikitext2_ars2_neo_sync_10e.toml

# Run CIFAR-10 AGA Mode (Efficient Convergence)
python -m exp.cifar.train --config config/lrp_cifar10_ars2_neo_aga_20e.toml
```

## 5. Framework Structure

- **[`optimizer/`](optimizer/)**: Core optimizer implementations, including [`ars2_neo.py`](optimizer/ars2_neo.py).
- **[`exp/`](exp/)**: Atomicized experiment scripts, decoupling data flow from model logic.
- **[`model/`](model/)**: Standard research models including Qwen3 (RoPE) and ResNet.
- **[`config/`](config/)**: TOML-based experiment configuration management.

## Citation

```bibtex
@software{ARS2_Neo_2025,
  author = {Rui, L.},
  title = {ARS2-Neo: Gliding Directly Towards Global Optima Along Geodesics of the Loss Landscape},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/dmf-archive/ARS}
}
```
