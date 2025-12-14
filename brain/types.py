from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np

@dataclass
class NeuromodulatoryState:
    dopamine_tonic: float = 0.2
    dopamine_phasic: float = 0.0
    serotonin: float = 0.5
    norepinephrine_tonic: float = 0.2
    norepinephrine_phasic: float = 0.0
    acetylcholine: float = 0.3
    cortisol: float = 0.0
    # Derived from LC
    attention_gain: float = 1.0
    arousal_boost: float = 0.0
    explore_gain: float = 1.0
    plasticity_gain: float = 1.0
    habit_weight: float = 1.0
    deliberation_weight: float = 1.0
    wm_stability: float = 1.0
    attention_breadth: float = 1.0
    ach_precision_gain: float = 0.0
    ach_mode: str = "BALANCED" # "ENCODING", "RECALL", or "BALANCED"

@dataclass
class SensoryState:
    vision_vec: np.ndarray = field(default_factory=lambda: np.zeros(2))
    touch: float = 0.0
    pain: float = 0.0
    novelty: float = 0.0
    vision_features: np.ndarray = field(default_factory=lambda: np.zeros(4)) # [wall, pred, targ, novelty]
    raw_vision_buffer: List[Dict[str, Any]] = field(default_factory=list)
    raw_whisker_hits: List[bool] = field(default_factory=list)
    raw_collision: bool = False

@dataclass
class MotorCommand:
    heading_vec: np.ndarray = field(default_factory=lambda: np.zeros(2))
    speed_multiplier: float = 1.0

@dataclass 
class CognitiveState:
    frustration: float = 0.0
    position: np.ndarray = field(default_factory=lambda: np.zeros(2))

@dataclass
class LearningSignals:
    reward: float = 0.0
    surprise: float = 0.0
    danger: float = 0.0
    prediction_error: float = 0.0 # This matches the 'pe_scalar' used in legacy code
