import numpy as np
from typing import Dict, Any
from ..interface import BrainComponent
from ..types import NeuromodulatoryState, SensoryState, CognitiveState, LearningSignals

def _ema(prev: float, target: float, tau: float, dt: float) -> float:
    alpha = 1.0 - np.exp(-dt / max(tau, 1e-9))
    return prev + alpha * (target - prev)

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

class BasalForebrainComponent(BrainComponent):
    @property
    def name(self) -> str:
        return "basal_forebrain"

    def __init__(self):
        self.tonic_ach = 0.2
        self.expected_uncertainty = 0.0
        self.mode = "BALANCED"

    def reset(self):
        self.tonic_ach = 0.2
        self.expected_uncertainty = 0.0
        self.mode = "BALANCED"

    def step(self, dt: float, neuromod: NeuromodulatoryState, sensory: SensoryState, cognitive: CognitiveState, learning: LearningSignals) -> Dict[str, Any]:
        # 1. Gather Inputs
        pe_scalar = _clamp(learning.prediction_error, 0.0, 1.0)
        
        # Reconstruct "Arousal" from available bus signals
        # (Legacy formula: phasic + frustration + touch + lc_boost + panic)
        # We use what we have on the bus:
        arousal = (neuromod.arousal_boost) + (cognitive.frustration * 0.5) + (sensory.touch * 0.1)
        arousal = _clamp(arousal, 0.0, 2.0)

        # 2. Update Uncertainty (Expected Uncertainty integrates PE variance)
        self.expected_uncertainty = _ema(self.expected_uncertainty, pe_scalar ** 2, tau=1.5, dt=dt)

        # 3. Calculate Tonic ACh
        target_ach = 0.2 + 0.4 * _clamp(arousal, 0.0, 1.5) - 0.5 * self.expected_uncertainty
        target_ach = _clamp(target_ach, 0.0, 1.0)
        self.tonic_ach = _ema(self.tonic_ach, target_ach, tau=1.0, dt=dt)

        # 4. Determine Mode & Precision
        if self.tonic_ach > 0.55:
            self.mode = "ENCODING"
        elif self.tonic_ach < 0.35:
            self.mode = "RECALL"
        else:
            self.mode = "BALANCED"

        precision_gain = _clamp(self.tonic_ach * 2.0, 0.0, 2.0)

        # 5. Output to Bus
        return {
            "neuromod.acetylcholine": self.tonic_ach,
            "neuromod.ach_precision_gain": precision_gain,
            "neuromod.ach_mode": self.mode
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "tonic_ach": self.tonic_ach,
            "mode": self.mode,
            "precision_gain": self.tonic_ach * 2.0,
            "expected_uncertainty": self.expected_uncertainty
        }
