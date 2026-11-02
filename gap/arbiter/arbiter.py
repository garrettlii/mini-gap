"""The learned arbiter: predicts each executor's success, picks the better one.

Features: node type, object pose relative to the gripper, instance perturbation
features, and the scripted skill's pre_margins(). Start with logistic regression
or gradient-boosted trees (sklearn). An explainable model beats a big one here:
the write-up should be able to say *when* the learned policy wins.
"""

from __future__ import annotations

import numpy as np


class Arbiter:
    def fit(self, X: np.ndarray, y_skill: np.ndarray, y_learned: np.ndarray) -> Arbiter:
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> tuple[float, float]:
        """(p_success_skill, p_success_learned)."""
        raise NotImplementedError

    def choose(self, x: np.ndarray, margin: float = 0.0) -> str:
        p_s, p_l = self.predict(x)
        return "learned" if p_l > p_s + margin else "skill"
