"""
QFA Inference Script — Lasso (Patch + LQR + Poly)
====================================================
Usage:
    1. Put submission_model.pkl in the same directory
    2. Prepare your data as X (n_samples, 4) and y (n_samples,)
    3. Call predict(X) or evaluate(X, y)

Example:
    from qfa_inference import QFAModel
    model = QFAModel("submission_model.pkl")

    # Predict on new data
    predictions = model.predict(X_new)

    # Or evaluate with known targets
    results = model.evaluate(X_new, y_true)
    print(results)
"""

import numpy as np
import pennylane as qml
import pickle
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr


class QFAModel:
    def __init__(self, pkl_path="submission_model.pkl"):
        with open(pkl_path, "rb") as f:
            bundle = pickle.load(f)

        self.model       = bundle["model"]
        self.scaler      = bundle["scaler"]
        self.poly        = bundle["poly"]
        self.patch_ry    = bundle["patch_ry"]
        self.patch_rz    = bundle["patch_rz"]
        self.lqr_rx      = bundle["lqr_rx"]
        self.lqr_ry      = bundle["lqr_ry"]
        self.lqr_rz      = bundle["lqr_rz"]
        self.bandwidths  = bundle["bandwidths"]
        self.patches     = bundle["patches"]
        self.patch_depth = bundle["patch_depth"]
        self.lqr_depth   = bundle["lqr_depth"]

        # Build circuits
        self.dev = qml.device("default.qubit", wires=2)
        self._patch_circuits = [self._make_patch_circuit(bw) for bw in self.bandwidths]
        self._lqr_circuits   = [self._make_lqr_circuit(bw) for bw in self.bandwidths]

        print(f"Model loaded: {np.sum(self.model.coef_ != 0)} active features")

    # ── Preprocessing ──────────────────────────────────────────────────

    def _preprocess(self, X_raw, clip_value=5.0):
        """
        Standardize, clip, and rescale raw features to [0, π].

        IMPORTANT: If your data is already in [0, π] (quantum-ready),
        pass it directly to predict_from_quantum() instead.
        """
        # Fit a fresh scaler on the provided data
        # For production use, you'd want to use the training scaler
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_raw)
        X_s = np.clip(X_s, -clip_value, clip_value)
        X_q = (X_s + clip_value) / (2 * clip_value) * np.pi
        return X_s, X_q

    def _preprocess_with_scaler(self, X_raw, fit_scaler, clip_value=5.0):
        """Use a provided scaler (e.g. fit on training data)."""
        X_s = fit_scaler.transform(X_raw)
        X_s = np.clip(X_s, -clip_value, clip_value)
        X_q = (X_s + clip_value) / (2 * clip_value) * np.pi
        return X_s, X_q

    # ── Circuit factories ──────────────────────────────────────────────

    def _make_patch_circuit(self, bandwidth):
        ry, rz, depth = self.patch_ry, self.patch_rz, self.patch_depth
        dev = self.dev

        @qml.qnode(dev)
        def circuit(x0, x1):
            for d in range(depth):
                qml.RY(ry[d, 0], wires=0)
                qml.RZ(rz[d, 0], wires=0)
                qml.RY(ry[d, 1], wires=1)
                qml.RZ(rz[d, 1], wires=1)
                qml.CZ(wires=[0, 1])
                qml.RZ(bandwidth * x0, wires=0)
                qml.RZ(bandwidth * x1, wires=1)
            return [
                qml.expval(qml.PauliX(0)), qml.expval(qml.PauliY(0)),
                qml.expval(qml.PauliZ(0)),
                qml.expval(qml.PauliX(1)), qml.expval(qml.PauliY(1)),
                qml.expval(qml.PauliZ(1)),
            ]
        return circuit

    def _make_lqr_circuit(self, bandwidth):
        rx, ry, rz = self.lqr_rx, self.lqr_ry, self.lqr_rz
        depth = self.lqr_depth
        dev = self.dev
        encode_gates = [qml.RX, qml.RY, qml.RZ]

        def _transform(x, d):
            mod = d % 3
            if mod == 0: return x
            elif mod == 1: return x * x
            else: return np.sin(x)

        @qml.qnode(dev)
        def circuit(x0, x1):
            for d in range(depth):
                qml.RX(rx[d, 0], wires=0)
                qml.RY(ry[d, 0], wires=0)
                qml.RZ(rz[d, 0], wires=0)
                qml.RX(rx[d, 1], wires=1)
                qml.RY(ry[d, 1], wires=1)
                qml.RZ(rz[d, 1], wires=1)
                qml.CZ(wires=[0, 1])
                gate = encode_gates[d % 3]
                gate(bandwidth * _transform(x0, d), wires=0)
                gate(bandwidth * _transform(x1, d), wires=1)
            return [
                qml.expval(qml.PauliX(0)), qml.expval(qml.PauliY(0)),
                qml.expval(qml.PauliZ(0)),
                qml.expval(qml.PauliX(1)), qml.expval(qml.PauliY(1)),
                qml.expval(qml.PauliZ(1)),
            ]
        return circuit

    # ── Feature extraction ─────────────────────────────────────────────

    def _extract_patch(self, X_q):
        out = []
        for idx, x in enumerate(X_q):
            if idx % 2000 == 0:
                print(f"  Patch: {idx}/{len(X_q)}")
            row = []
            for c in self._patch_circuits:
                for (i, j) in self.patches:
                    row.extend(np.array(c(x[i], x[j])))
            out.append(row)
        return np.array(out)

    def _extract_lqr(self, X_q):
        out = []
        for idx, x in enumerate(X_q):
            if idx % 2000 == 0:
                print(f"  LQR: {idx}/{len(X_q)}")
            row = []
            for c in self._lqr_circuits:
                for (i, j) in self.patches:
                    row.extend(np.array(c(x[i], x[j])))
            out.append(row)
        return np.array(out)

    def extract_features(self, X_s, X_q):
        """
        Extract all features from standardized (X_s) and quantum-ready (X_q) data.
        Returns the full 230-dim feature vector, standardized and ready for prediction.
        """
        print("Extracting patch features...")
        F_patch = self._extract_patch(X_q)

        print("Extracting LQR features...")
        F_lqr = self._extract_lqr(X_q)

        print("Computing polynomial features...")
        F_poly = self.poly.transform(X_s)

        X_all = np.hstack([F_patch, F_lqr, F_poly])
        X_all_s = self.scaler.transform(X_all)
        return X_all_s

    # ── Prediction ─────────────────────────────────────────────────────

    def predict(self, X_raw, preprocess_scaler=None):
        """
        End-to-end: raw features (n_samples, 4) → predictions.

        Args:
            X_raw: array of shape (n_samples, 4)
            preprocess_scaler: optional fitted StandardScaler from training data.
                If None, fits a new scaler on X_raw (fine for large datasets,
                less ideal for small batches).
        """
        if preprocess_scaler is not None:
            X_s, X_q = self._preprocess_with_scaler(X_raw, preprocess_scaler)
        else:
            X_s, X_q = self._preprocess(X_raw)

        X_features = self.extract_features(X_s, X_q)
        return self.model.predict(X_features)

    def predict_from_quantum(self, X_s, X_q):
        """
        Predict from already-preprocessed data.
        X_s: standardized features (for poly)
        X_q: quantum-ready features in [0, π] (for circuits)
        """
        X_features = self.extract_features(X_s, X_q)
        return self.model.predict(X_features)

    # ── Evaluation ─────────────────────────────────────────────────────

    def evaluate(self, X_raw_or_s, y_true, X_q=None, preprocess_scaler=None):
        """
        Predict and compute metrics.

        Two calling conventions:
            evaluate(X_raw, y_true)                    — from raw features
            evaluate(X_s, y_true, X_q=X_q)             — from preprocessed
        """
        if X_q is not None:
            preds = self.predict_from_quantum(X_raw_or_s, X_q)
        else:
            preds = self.predict(X_raw_or_s, preprocess_scaler)

        mse  = mean_squared_error(y_true, preds)
        mae  = mean_absolute_error(y_true, preds)
        corr = pearsonr(y_true, preds)[0]
        return {"MSE": round(mse, 6), "MAE": round(mae, 6),
                "Corr": round(corr, 4), "n_samples": len(y_true)}


# ── Standalone usage ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading model...")
    model = QFAModel("submission_model.pkl")

    print("\nTo use in your notebook:")
    print("  from qfa_inference import QFAModel")
    print("  model = QFAModel('submission_model.pkl')")
    print()
    print("  # From raw data:")
    print("  preds = model.predict(X_new)")
    print()
    print("  # From preprocessed data:")
    print("  preds = model.predict_from_quantum(X_s, X_q)")
    print()
    print("  # Evaluate with targets:")
    print("  results = model.evaluate(X_s, y_true, X_q=X_q)")
