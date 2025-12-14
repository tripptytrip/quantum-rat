import numpy as np
from typing import Dict, Any
from ..interface import BrainComponent
from ..types import NeuromodulatoryState, SensoryState, CognitiveState, LearningSignals

def _ema(prev: float, target: float, tau: float, dt: float) -> float:
    alpha = 1.0 - np.exp(-dt / max(tau, 1e-9))
    return prev + alpha * (target - prev)

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

class LocusCoeruleusComponent(BrainComponent):
    @property
    def name(self) -> str:
        return "locus_coeruleus"

    def __init__(self):
        self.tonic_ne = 0.2
        self.phasic_ne = 0.0
        self.uncertainty = 0.0
        self.reward_pred = 0.0
        self.mode = "EXPLOIT"

    def reset(self):
        self.tonic_ne = 0.2
        self.phasic_ne = 0.0
        self.uncertainty = 0.0
        self.reward_pred = 0.0
        self.mode = "EXPLOIT"

    def step(self, dt: float, neuromod: NeuromodulatoryState, sensory: SensoryState, cognitive: CognitiveState, learning: LearningSignals) -> Dict[str, Any]:
        # Logic copied from original app.py LocusCoeruleus
        reward = learning.reward
        novelty = sensory.novelty
        pain = sensory.pain
        danger = learning.danger
        surprise = learning.surprise

        # 1. Uncertainty
        reward_surprise = abs(reward - self.reward_pred)
        self.reward_pred = _ema(self.reward_pred, reward, tau=0.25, dt=dt)
        unc_input = max(reward_surprise, surprise)
        self.uncertainty = _ema(self.uncertainty, unc_input, tau=2.0, dt=dt)

        # 2. Phasic NE
        salience = max(novelty, pain, danger, reward_surprise, surprise)
        self.phasic_ne = _clamp(_ema(self.phasic_ne, salience, tau=0.10, dt=dt), 0.0, 1.0)

        # 3. Tonic NE
        target_tonic = 0.15 + (0.6 * self.uncertainty)
        target_tonic += 0.3 * novelty
        target_tonic += 0.25 * danger
        target_tonic += 0.15 * pain
        target_tonic = _clamp(target_tonic, 0.0, 1.0)
        self.tonic_ne = _clamp(_ema(self.tonic_ne, target_tonic, tau=1.50, dt=dt), 0.0, 1.0)

        # 4. Mode
        if self.tonic_ne < 0.35: self.mode = "EXPLOIT"
        elif self.tonic_ne < 0.65: self.mode = "BALANCED"
        else: self.mode = "EXPLORE"

        # 5. Outputs
        habit_weight = _clamp(1.2 - (self.tonic_ne * 1.0), 0.2, 1.2)
        deliberation_weight = _clamp(0.5 + (self.tonic_ne * 1.0), 0.5, 1.5)
        
        if self.phasic_ne < 0.5:
            wm_stability = 1.0
        else:
            wm_stability = _clamp(1.0 - (self.phasic_ne - 0.5) * 1.4, 0.3, 1.0)

        optimal_ne = 0.5
        distance = abs(self.tonic_ne - optimal_ne)
        attention_gain = _clamp(1.5 - (distance * 1.0), 0.7, 1.5)
        attention_gain += self.phasic_ne * 0.4
        attention_gain = _clamp(attention_gain, 0.5, 2.0)

        attention_breadth = _clamp(0.3 + (self.tonic_ne * 1.2), 0.3, 1.5)
        explore_gain = _clamp(0.6 + (self.tonic_ne * 1.8), 0.2, 3.0)
        plasticity_gain = _clamp(0.8 + (self.phasic_ne * 1.0), 0.5, 2.0)
        arousal_boost = (0.6 * self.tonic_ne) + (1.0 * self.phasic_ne)

        print(f"LC_NEW: tonic={self.tonic_ne:.3f} phasic={self.phasic_ne:.3f}")
        return {
            "neuromod.norepinephrine_tonic": self.tonic_ne,
            "neuromod.norepinephrine_phasic": self.phasic_ne,
            "neuromod.attention_gain": attention_gain,
            "neuromod.arousal_boost": arousal_boost,
            "neuromod.explore_gain": explore_gain,
            "neuromod.plasticity_gain": plasticity_gain,
            "neuromod.habit_weight": habit_weight,
            "neuromod.deliberation_weight": deliberation_weight,
            "neuromod.wm_stability": wm_stability,
            "neuromod.attention_breadth": attention_breadth,
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "tonic_ne": self.tonic_ne,
            "phasic_ne": self.phasic_ne,
            "uncertainty": self.uncertainty,
            "mode": self.mode,
            # Pass through the last calculated outputs for UI visualization
            "habit_weight": 1.2 - (self.tonic_ne * 1.0), # Approximate reconstruction or store in self.last
            "wm_stability": 1.0 if self.phasic_ne < 0.5 else (1.0 - (self.phasic_ne - 0.5) * 1.4)
        }
