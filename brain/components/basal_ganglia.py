import numpy as np
from typing import Dict, Any, Tuple
from ..interface import BrainComponent
from ..types import NeuromodulatoryState, SensoryState, CognitiveState, LearningSignals


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class PVLV_Learning:
    """
    Primary Value / Learned Value model for reward prediction.
    Produces dopamine signals based on reward prediction error.
    """
    
    def __init__(self):
        self.w_pv = 0.0  # Primary value weight
        self.w_lv = 0.0  # Learned value weight
        self.alpha_pv = 0.1
        self.alpha_lv = 0.05

    def step(self, sensory_drive: float, reward_present: float, 
             baseline_dopamine: float = 0.0, serotonin: float = 0.5) -> Tuple[float, float]:
        sensory_drive = float(max(0.0, sensory_drive))
        reward_present = float(max(0.0, reward_present))
        baseline_dopamine = float(_clamp(baseline_dopamine, 0.0, 1.5))
        serotonin = float(_clamp(serotonin, 0.0, 1.0))

        # Patience factor: high serotonin = more patient, discounts less
        patience_factor = 0.5 + (serotonin * 0.5)
        discounted_reward = reward_present * patience_factor

        # Primary Value (PV): learns direct reward prediction
        pv_prediction = self.w_pv * sensory_drive
        pv_error = discounted_reward - pv_prediction
        self.w_pv += self.alpha_pv * pv_error * sensory_drive
        self.w_pv = _clamp(self.w_pv, -2.0, 2.0)  # Prevent runaway

        # Learned Value (LV): learns to predict PV
        lv_prediction = self.w_lv * sensory_drive
        lv_error = pv_prediction - lv_prediction
        self.w_lv += self.alpha_lv * lv_error * sensory_drive
        self.w_lv = _clamp(self.w_lv, -2.0, 2.0)

        phasic_dopamine = pv_error + lv_prediction
        dopamine_total = _clamp(baseline_dopamine + phasic_dopamine, -1.0, 2.0)
        
        return dopamine_total, phasic_dopamine


class BasalGangliaComponent(BrainComponent):
    """
    Basal Ganglia: Action selection, gating, and reinforcement learning.
    
    Integrates dopamine (PVLV), serotonin (patience), and LC signals
    to select between sensory-driven and memory-driven behavior.
    """
    
    @property
    def name(self) -> str:
        return "basal_ganglia"

    def __init__(self, rng):
        self.rng = rng
        self.reset()

    def reset(self):
        self.w_gate = self.rng.random(3)  # Weights for gating [sensory, memory, drive]
        self.w_motor = np.ones(3)  # Weights for action [sensory_path, memory_path, explore]
        self.pvlv = PVLV_Learning()
        
        # Diagnostics
        self.last_gate_signal = 0.0
        self.last_dopa_total = 0.0
        self.last_dopa_phasic = 0.0

    def step(self, dt: float, neuromod: NeuromodulatoryState, sensory: SensoryState, 
             cognitive: CognitiveState, learning: LearningSignals) -> Dict[str, Any]:
        
        # === INPUTS ===
        sensory_vec = sensory.vision_vec
        memory_vec = cognitive.working_memory
        
        sensory_mag = float(np.linalg.norm(sensory_vec))
        memory_mag = float(np.linalg.norm(memory_vec))
        
        # === DOPAMINE (PVLV) ===
        dopa_total, dopa_phasic = self.pvlv.step(
            sensory_drive=sensory_mag,
            reward_present=learning.reward,
            baseline_dopamine=neuromod.dopamine_tonic,
            serotonin=neuromod.serotonin
        )
        
        # === GATING DECISION ===
        # Context: [sensory_intensity, memory_intensity, frustration]
        context = np.array([sensory_mag, memory_mag, cognitive.frustration], dtype=float)
        
        gate_activation = float(np.dot(self.w_gate, context) + dopa_total)
        gate_threshold = 0.5
        gate_signal = 1.0 if gate_activation > gate_threshold else -1.0
        
        # === ACTION SELECTION ===
        serotonin = _clamp(neuromod.serotonin, 0.0, 1.0)
        
        # High serotonin = patience (favor memory/planning)
        # Low serotonin = impulsivity (favor sensory/reflex)
        sensory_bias = 1.0 - serotonin * 0.3
        memory_bias = 0.5 + serotonin * 0.5
        
        habit_weight = neuromod.habit_weight  # From LC
        
        w_sensory = self.w_motor[0] * (1.0 + max(0, dopa_total)) * sensory_bias * habit_weight
        w_memory = self.w_motor[1] * (1.0 + max(0, -dopa_total) * 0.3) * memory_bias
        
        motor_out = (sensory_vec * w_sensory) + (memory_vec * w_memory)
        
        # === EXPLORATORY NOISE ===
        # When explore_gain is high (LC in EXPLORE mode), add movement noise
        explore_gain = neuromod.explore_gain
        if explore_gain > 1.2:
            noise = self.rng.normal(0.0, 1.0, size=2)
            noise_scale = (explore_gain - 1.0) * 0.4
            motor_out = motor_out + noise * noise_scale

        # === LEARNING ===
        if neuromod.plasticity_gain > 0.1 and abs(dopa_phasic) > 0.05:
            lr = 0.01 * neuromod.plasticity_gain
            
            # Hebbian update: strengthen active pathways when rewarded
            # Sign of dopa_phasic determines reinforcement vs punishment
            delta_gate = lr * dopa_phasic * context
            delta_motor = lr * dopa_phasic * np.array([sensory_mag, memory_mag, 1.0])
            
            self.w_gate = np.clip(self.w_gate + delta_gate, -2.0, 2.0)
            self.w_motor = np.clip(self.w_motor + delta_motor, 0.1, 3.0)

        # === DIAGNOSTICS ===
        self.last_gate_signal = gate_signal
        self.last_dopa_total = dopa_total
        self.last_dopa_phasic = dopa_phasic

        return {
            "motor.heading_vec": motor_out,
            "cognitive.gate_signal": gate_signal,
            "neuromod.dopamine_phasic": dopa_phasic
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "gate_signal": self.last_gate_signal,
            "dopamine_total": self.last_dopa_total,
            "dopamine_phasic": self.last_dopa_phasic,
            "w_gate": self.w_gate.tolist(),
            "w_motor": self.w_motor.tolist(),
            "pvlv_w_pv": self.pvlv.w_pv,
            "pvlv_w_lv": self.pvlv.w_lv
        }
