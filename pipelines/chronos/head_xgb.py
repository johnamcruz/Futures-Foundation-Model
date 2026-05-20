"""Generic XGBoost classifier head — the trained component.

Strategy-agnostic: consumes a feature matrix X (frozen Chronos embedding,
optionally fused with the strategy's own features) and integer labels;
emits a class per row plus class probabilities (the conviction signal a
downstream gate / account layer consumes). Deterministic given `seed`
(single-threaded, fixed random_state). ONNX-convertible (skl2onnx) — see
the deployment memory; nothing here blocks the single-file export.
"""
import numpy as np


class XGBHead:
    """Head with fit(X,y,seed) / predict(X) / predict_proba(X).
    Conforms to the head-agnostic contract evaluate.py speaks."""

    def __init__(self, n_classes, n_estimators=200, max_depth=4,
                 learning_rate=0.05, subsample=0.8, colsample_bytree=0.8):
        self.n_classes = int(n_classes)
        self._p = dict(n_estimators=n_estimators, max_depth=max_depth,
                       learning_rate=learning_rate, subsample=subsample,
                       colsample_bytree=colsample_bytree)
        self._clf = None

    def fit(self, X, y, seed=0):
        # Let XGBClassifier infer the objective from y: binary:logistic for
        # 2 classes, multi:softprob for >2. Forcing multi+num_class breaks
        # the binary case (predict returns 2-D).
        import xgboost as xgb
        self._clf = xgb.XGBClassifier(
            tree_method='hist', random_state=seed, n_jobs=1,
            verbosity=0, **self._p)
        self._clf.fit(np.asarray(X, np.float32), np.asarray(y))
        return self

    def predict(self, X):
        return self._clf.predict(np.asarray(X, np.float32)).astype(int)

    def predict_proba(self, X):
        return self._clf.predict_proba(np.asarray(X, np.float32))


class XGBRiskHead:
    """Regression head — predicts max_rr_realized per signal (the peak R
    reached before stop / vertical barrier). Used at inference to set
    dynamic TP per trade (TP = clip(0.8 * R_hat, 1.5, 8.0)). Mirrors the
    FFM risk-head design: Huber loss (less outlier-sensitive than MSE),
    same input features as the signal head, ONNX-convertible head."""

    def __init__(self, n_estimators=300, max_depth=4, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8):
        self._p = dict(n_estimators=n_estimators, max_depth=max_depth,
                       learning_rate=learning_rate, subsample=subsample,
                       colsample_bytree=colsample_bytree)
        self._reg = None

    def fit(self, X, y_max_rr, seed=0):
        # FFM uses Huber/SmoothL1; XGBoost has pseudohubererror which is
        # twice-differentiable and equivalent in spirit.
        import xgboost as xgb
        self._reg = xgb.XGBRegressor(
            objective='reg:pseudohubererror', huber_slope=1.0,
            tree_method='hist', random_state=seed, n_jobs=1,
            verbosity=0, **self._p)
        self._reg.fit(np.asarray(X, np.float32),
                      np.asarray(y_max_rr, np.float32))
        return self

    def predict(self, X):
        # Clip to plausible R range to silence absurd extrapolations.
        return np.clip(
            self._reg.predict(np.asarray(X, np.float32)).astype(np.float32),
            0.0, 15.0)
