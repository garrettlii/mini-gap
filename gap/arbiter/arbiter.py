"""The learned arbiter: predicts each executor's success, picks the better one.

Features: what the hand sees of the object when the node starts (fallback.observe: offsets
along the closing and width axes, height, yaw off square, object half height, roundness),
as absolute values and squares. One logistic regression per executor, so the write-up can
say *when* the learned policy wins from the coefficients.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ("|close_axis|", "|width_axis|", "|dz|", "|yaw|", "half", "round",
            "close_axis^2", "width_axis^2", "seen")   # seen: wrist observations only


def featurize(obs: np.ndarray) -> np.ndarray:
    """fallback.observe() vector(s) -> arbiter features (rows)."""
    obs = np.atleast_2d(obs)
    ca, wa, dz, yaw, half, round_ = obs[:, :6].T
    F = np.c_[np.abs(ca), np.abs(wa), np.abs(dz), np.abs(yaw), half, round_, ca**2, wa**2]
    return np.c_[F, obs[:, 8]] if obs.shape[1] > 8 else F


class Arbiter:
    def __init__(self):
        self.models = {k: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
                       for k in ("skill", "learned")}

    def fit(self, X: np.ndarray, y_skill: np.ndarray, y_learned: np.ndarray) -> Arbiter:
        """X: rows of fallback.observe(); labels: did each executor pass the node's post-check."""
        F = featurize(X)
        for key, y in (("skill", y_skill), ("learned", y_learned)):
            y = np.asarray(y, dtype=int)
            if len(set(y)) < 2:   # one class only: predict that rate
                self.models[key] = float(y.mean())
            else:
                self.models[key].fit(F, y)
        return self

    def predict(self, x: np.ndarray) -> tuple[float, float]:
        """(p_success_skill, p_success_learned)."""
        F = featurize(x)
        p = [m if isinstance(m, float) else float(m.predict_proba(F)[0, 1])
             for m in (self.models["skill"], self.models["learned"])]
        return p[0], p[1]

    def choose(self, x: np.ndarray, margin: float = 0.0) -> str:
        p_s, p_l = self.predict(x)
        return "learned" if p_l > p_s + margin else "skill"
