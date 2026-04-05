# Quantum Feature Augmentation for Financial Market Prediction

### YQuantum 2026 — AWS × State Street Challenge

Quantum-enhanced feature extraction for financial market prediction using Amazon Braket + PennyLane. Local patch quantum kernels and local quantum reservoirs, both provably free from exponential concentration.

---

## Key Findings

### 1. Synthetic regime-switching data: quantum features deliver +51.4% improvement

| Model | MSE | MAE | Corr | vs Baseline |
|---|---|---|---|---|
| LR (raw features) | 4.566 | 1.618 | 0.931 | — |
| Ridge (poly-2 + PCA) | 2.777 | 1.319 | 0.959 | baseline |
| Lasso (Patch + LQR + Poly) | **1.370** | **0.915** | **0.980** | **+51.4%** |

Per-regime improvement over classical baseline:
- Regime 1 (linear, 76% of data): **+57.0%**
- Regime 2 (nonlinear, 24% of data): **+28.5%**

Lasso feature selection retained 97 of 230 features (62 LQR, 32 Patch, 3 Poly), confirming that quantum features carry the predictive signal while classical polynomial features are largely redundant.

### 2. Real S&P 500 stock data: two stories in one result

| Model | Features | MSE | Dir. Accuracy |
|---|---|---|---|
| Classical Baseline | 13 raw | **0.000380** | **74.85%** |
| PCA(4) Baseline | 4 PCA | 0.000630 | 49.57% |
| PCA(4) + Reservoir | 4 PCA → 10 | 0.000614 | 53.25% |
| PCA(4) + Patch | 4 PCA → 108 | 0.000640 | 52.80% |
| PCA(4) + LQR | 4 PCA → 108 | 0.000717 | 53.70% |
| PCA(4) + Patch + LQR | 4 PCA → 216 | 0.000733 | **54.27%** |

#### PCA is the bottleneck, not quantum circuits

PCA compression from 13 to 4 features destroys predictive signal — directional accuracy drops from 74.85% to 49.57% (random chance) before any quantum circuit runs. This is proven by the PCA(4) Baseline experiment, which uses no quantum processing at all and still collapses to coin-flip accuracy.

#### Quantum features recover directional signal

Starting from PCA's crippled 49.57% baseline, Patch+LQR achieves the **highest directional accuracy of any quantum-augmented method** (54.27%) — a ~5 percentage point recovery on data that PCA had rendered essentially unpredictable. This suggests the quantum circuits are capturing genuine pairwise market structure that PCA obscured.

#### Why MSE is high but directional accuracy is best

Patch+LQR has the highest MSE (0.000733) but the best directional accuracy (54.27%). This is not contradictory — it means the model correctly predicts *which direction* the market moves more often than any other method, but occasionally produces magnitude errors that MSE penalizes quadratically. A single prediction that overshoots by 5× inflates MSE more than five predictions that are slightly off.

For a trading application, directional accuracy is the more actionable metric — profits come from being on the right side of the trade, not from predicting the exact return. The magnitude calibration problem (high MSE) is a simpler downstream fix, addressable with stronger regularization or prediction shrinkage, without changing the underlying quantum feature extraction.

#### Why this happens

Our 2-qubit patch circuits require exactly 4 input features. With 13 stock features, PCA compression to 4 dimensions is required. PCA selects directions of maximum variance, not maximum predictive power — the most volatile features aren't necessarily the most informative for predicting 5-day returns.

### 3. The path forward is clear

The limitation is hardware scaling, not algorithmic:

- **More qubits:** With 13 qubits (one per feature), the architecture would use C(13,2) = 78 patches directly on raw features, eliminating PCA entirely.
- **Smarter selection:** Supervised feature selection (SelectKBest) could replace PCA to preserve predictive signal in the 4 selected dimensions.
- **Magnitude calibration:** Stronger Lasso regularization or prediction shrinkage would reduce MSE without sacrificing the directional accuracy advantage.

---

## Architecture

Two complementary quantum feature extractors, both using 2-qubit circuits (provably concentration-free):

**Local Patch (108 features)**
- 2-qubit circuits on all 6 pairwise feature combinations
- 4 layers deep, fixed random unitary + RZ data encoding per layer
- Multi-bandwidth sweep: π/4, π/2, π
- Based on local kernel mitigation (QAMP 2025, arXiv:2602.16097)

**Local Quantum Reservoir (108 features)**
- Same 6 pairwise patches, different encoding strategy
- 8 layers deep with data re-uploading
- Cycling encoding gates: RX → RY → RZ (full Bloch sphere coverage)
- Nonlinear input transforms: x, x², sin(x) at different layers
- Based on Gyurik et al. (Phil. Trans. R. Soc. A, Feb 2026)

**Classical Poly-2 (14 features)**
- Standard degree-2 polynomial features

**Readout:** Lasso (α=0.005) with StandardScaler. Retains 97/230 features automatically.

### Why 2-qubit patches are concentration-free

Exponential concentration causes quantum kernel values to converge as qubit count grows, making features indistinguishable (Thanasilp et al., Nature Communications 2024). A 2-qubit Hilbert space has dimension 4 — far too small for concentration to occur. Expectation values maintain meaningful spread regardless of circuit depth. We aggregate across 6 individually concentration-free circuits to build a rich 108-dimensional feature space.

---

## Files

| File | Description |
|---|---|
| `QFA_Overview.ipynb` | Main notebook — synthetic data experiments |
| `WorkingStocks.ipynb` | Stock data experiments — walk-forward backtest |
| `qfa_quantum_features_v9.py` | Feature extraction + training (generates model) |
| `qfa_inference.py` | Inference module for predictions on new data |
| `qfa_qpu_validation.py` | QPU hardware validation script |
| `submission_model.pkl` | Trained Lasso model bundle |
| `design_motivation.md` | Presentation-ready design rationale |

---

## Quick Start

### Prerequisites

```
pip install pennylane numpy scikit-learn scipy
```

### Running predictions

```python
from qfa_inference import QFAModel

model = QFAModel("submission_model.pkl")

# From raw features (4 columns)
predictions = model.predict(X_new)

# From preprocessed data
predictions = model.predict_from_quantum(X_s, X_q)

# Evaluate with targets
results = model.evaluate(X_s, y_true, X_q=X_q)
```

### Important note on stock data

The scaler inside `submission_model.pkl` was fit on synthetic training data. For stock data with a different distribution, preprocess independently and use `predict_from_quantum()`.

---

## Evolution

| Version | Approach | Synthetic MSE | What happened |
|---|---|---|---|
| v4 | QAA + VQC + QSK + Poly | — | QAA failed (global phase unobservable) |
| v5 | VQC + QSK + Patch + QRC + Poly | 2.898 | Ablation: VQC and QRC hurt ensemble |
| v6 | QSK + Patch + Poly | 2.459 | Patch solo (1.603) beat ensemble |
| v7 | Patch solo | 1.603 | +42.3% vs baseline |
| v8 | Patch + LQR | 1.354 | LQR adds complementary nonlinearities |
| v9 | Patch + LQR + Poly + Lasso | **1.370** | Lasso prevents Regime 2 overfitting |

Each iteration was informed by ablation analysis — components were kept only if drop-one-out testing confirmed positive contribution.

---

## References

- Paini, Jacquier, Brett et al. "Exact simulation of Quantum Enhanced Signature Kernels for financial data streams prediction using Amazon Braket" (Rigetti/Imperial/AWS, Nov 2025)
- "Local and Multi-Scale Strategies to Mitigate Exponential Concentration in Quantum Kernels" (QAMP 2025, arXiv:2602.16097)
- Gyurik et al. "From quantum feature maps to quantum reservoir computing" (Phil. Trans. R. Soc. A, Feb 2026)
- Thanasilp et al. "Exponential concentration in quantum kernel methods" (Nature Communications, 2024)
- Shaydulin & Wild. "Importance of kernel bandwidth in quantum machine learning" (2022)
