"""Learned fallback executors, one per manipulation node type.

Behavior cloning on oracle rollouts (dev or rehearsal seeds only).
Start: privileged-state input, sklearn MLPRegressor predicting the next joint-target
delta, run closed-loop at a fixed control rate. This trains on a laptop CPU.
Stretch: image input (cloud GPU), or a fine-tuned VLA as the learned executor.
"""

from __future__ import annotations

import numpy as np

NODE_TYPES = ("reach", "grasp", "place")


class FallbackPolicy:
    def __init__(self, node_type: str):
        assert node_type in NODE_TYPES
        self.node_type = node_type

    def featurize(self, state: dict, goal: dict) -> np.ndarray:
        """Goal-relative features: object pose minus TCP pose, gripper width, joint angles."""
        raise NotImplementedError

    def fit(self, X: np.ndarray, Y: np.ndarray) -> FallbackPolicy:
        raise NotImplementedError

    def act(self, state: dict, goal: dict) -> np.ndarray:
        raise NotImplementedError
