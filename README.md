# Quantum Feature Augmentation — YQuantum 2026

Quantum-enhanced feature extraction for financial market prediction using Amazon Braket + PennyLane.

## Results

| Model | MSE | MAE | Corr | vs Baseline |
|---|---|---|---|---|
| Ridge (poly-2 + PCA) | 2.777 | 1.319 | 0.9585 | — |
| Lasso (Patch + LQR + Poly) | **1.370** | **0.915** | **0.9798** | **+51.4%** |

Per-regime improvement over classical baseline:
- Regime 1 (linear, 76% of data): +57.0%
- Regime 2 (nonlinear, 24% of data): +28.5%

## Files

| File | Description |
|---|---|
| `QFA_Overview.ipynb` | Main notebook (data gen, baselines, all experiments) |
| `qfa_quantum_features_v9.py` | Feature extraction + training script (generates the model) |
| `qfa_inference.py` | Inference module for running predictions on new data |
| `submission_model.pkl` | Trained model bundle (Lasso + scaler + circuit weights) |

## Quick start

### Prerequisites

```
pip install pennylane numpy scikit-learn scipy
```

### Running predictions on new data

```python
from qfa_inference import QFAModel

model = QFAModel("submission_model.pkl")
```

**From raw features** (4 columns, any scale — preprocessing handled internally):

```python
predictions = model.predict(X_stock)
```

**From preprocessed data** (if you already standardized and rescaled to [0, π]):

```python
predictions = model.predict_from_quantum(X_stock_s, X_stock_q)
```

**Evaluate against known targets:**

```python
results = model.evaluate(X_stock_s, y_stock, X_q=X_stock_q)
print(results)
# {'MSE': ..., 'MAE': ..., 'Corr': ..., 'n_samples': ...}
```

### Important note on stock data

The scaler inside `submission_model.pkl` was fit on synthetic training data. For real stock data with a different distribution, either:

1. Pass your own scaler: `model.predict(X_raw, preprocess_scaler=your_scaler)`
2. Preprocess yourself and use: `model.predict_from_quantum(X_s, X_q)`

Feature extraction takes ~15–20 min per 10k samples (quantum circuits must run). The Lasso prediction itself is instant after that.

## Architecture

Two complementary quantum feature extractors, both using 2-qubit circuits (concentration-free by construction):

**Local Patch (108 features)** — 2-qubit circuits on all 6 pairwise feature combinations, 4 layers deep, RZ data encoding, swept across 3 bandwidths (π/4, π/2, π). Based on the local kernel mitigation strategy from QAMP 2025.

**Local Quantum Reservoir (108 features)** — Same 6 patches but with reservoir-style encoding: 8 layers deep, cycling encoding gates (RX → RY → RZ) with nonlinear input transforms (x, x², sin(x)). Based on Gyurik et al., "From quantum feature maps to quantum reservoir computing" (2026).

**Classical Poly-2 (14 features)** — Standard degree-2 polynomial features.

Total: 230 features → StandardScaler → Lasso (α=0.005) → 97 features kept.

## Evolution

| Version | Approach | MSE | What happened |
|---|---|---|---|
| v4 | QAA + VQC + QSK + Poly | — | QAA failed (global phase unobservable) |
| v5 | VQC + QSK + Patch + QRC + Poly | 2.898 | Ablation: VQC and QRC hurt ensemble |
| v6 | QSK + Patch + Poly | 2.459 | Patch solo (1.603) beat ensemble |
| v7 | Patch solo | 1.603 | +42.3% vs baseline, clean model |
| v8 | Patch + LQR | 1.354 | LQR adds complementary nonlinearities |
| v9 | Patch + LQR + Poly + Lasso | **1.370** | Lasso prevents Regime 2 overfitting |

## References

- Paini, Jacquier, Brett et al. "Exact simulation of Quantum Enhanced Signature Kernels for financial data streams prediction using Amazon Braket" (Rigetti/Imperial/AWS, Nov 2025)
- "Local and Multi-Scale Strategies to Mitigate Exponential Concentration in Quantum Kernels" (QAMP 2025, arXiv:2602.16097)
- Gyurik et al. "From quantum feature maps to quantum reservoir computing" (Phil. Trans. R. Soc. A, Feb 2026)
- Shaydulin & Wild. "Importance of kernel bandwidth in quantum machine learning" (2022)
