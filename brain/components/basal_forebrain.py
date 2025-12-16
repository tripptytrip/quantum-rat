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
    """
    Basal Forebrain: Acetylcholine system controlling precision and encoding/recall modes.
    
    High ACh = high precision, encoding mode (attend to sensory input)
    Low ACh = low precision, recall mode (rely on memory/priors)
    """
    
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

    def step(self, dt: float, neuromod: NeuromodulatoryState, sensory: SensoryState, 
             cognitive: CognitiveState, learning: LearningSignals) -> Dict[str, Any]:
        
        # === INPUTS ===
        pe_scalar = _clamp(learning.prediction_error, 0.0, 1.0)
        
        # Arousal approximation from available signals
        arousal = neuromod.arousal_boost + (cognitive.frustration * 0.5) + (sensory.touch * 0.1)
        arousal = _clamp(arousal, 0.0, 2.0)

        # === EXPECTED UNCERTAINTY ===
        # Integrates prediction error variance over time
        self.expected_uncertainty = _ema(self.expected_uncertainty, pe_scalar ** 2, tau=1.5, dt=dt)

        # === TONIC ACh ===
        # High arousal → high ACh (attend to world)
        # High expected uncertainty → low ACh (rely on priors, world is unpredictable)
        target_ach = 0.2 + 0.4 * _clamp(arousal, 0.0, 1.5) - 0.5 * self.expected_uncertainty
        target_ach = _clamp(target_ach, 0.0, 1.0)
        self.tonic_ach = _ema(self.tonic_ach, target_ach, tau=1.0, dt=dt)

        # === MODE ===
        if self.tonic_ach > 0.55:
            self.mode = "ENCODING"
        elif self.tonic_ach < 0.35:
            self.mode = "RECALL"
        else:
            self.mode = "BALANCED"

        precision_gain = _clamp(self.tonic_ach * 2.0, 0.0, 2.0)

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
