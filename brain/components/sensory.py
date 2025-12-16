import numpy as np
from typing import Dict, Any, List
from ..interface import BrainComponent
from ..types import NeuromodulatoryState, SensoryState, CognitiveState, LearningSignals

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class SomatoCortexComponent(BrainComponent):
    @property
    def name(self) -> str:
        return "somato_cortex"

    def __init__(self):
        self.ema_pain = 0.0

    def reset(self):
        self.ema_pain = 0.0

    def step(self, dt: float, neuromod: NeuromodulatoryState, sensory: SensoryState, 
             cognitive: CognitiveState, learning: LearningSignals) -> Dict[str, Any]:
        hits = sensory.raw_whisker_hits
        collision = sensory.raw_collision
        
        touch = 1.0 if hits and any(hits) else 0.0
        pain = 1.0 if collision else 0.0
        
        # Decay/Smooth pain signal
        self.ema_pain = max(pain, self.ema_pain * 0.9)
        
        return {
            "sensory.touch": touch,
            "sensory.pain": self.ema_pain
        }


class VisionCortexComponent(BrainComponent):
    @property
    def name(self) -> str:
        return "vision_cortex"

    def __init__(self):
        self.ema_salience = 0.0

    def reset(self):
        self.ema_salience = 0.0

    def step(self, dt: float, neuromod: NeuromodulatoryState, sensory: SensoryState, 
             cognitive: CognitiveState, learning: LearningSignals) -> Dict[str, Any]:
        vision_rays = sensory.raw_vision_buffer
        
        # Salience weights by object type
        salience_weights = {1: 0.5, 2: 2.0, 3: 1.5}  # wall, predator, target
        
        max_salience = 0.0
        max_target_salience = 0.0
        salient_target_ray = None

        wall_dist = 15.0
        predator_dist = 15.0
        target_dist = 15.0

        for ray in vision_rays:
            dist = ray.get("dist", 15.0)
            typ = ray.get("type", 0)
            
            # Track minimum distances by type
            if typ == 1: wall_dist = min(wall_dist, dist)
            if typ == 2: predator_dist = min(predator_dist, dist)
            if typ == 3: target_dist = min(target_dist, dist)

            # Salience calculation with saturation to prevent explosion at close range
            if typ in salience_weights:
                # Saturating salience: max out at ~2.0 when very close
                normalized_dist = dist / 15.0  # 0 = touching, 1 = max range
                proximity = 1.0 - normalized_dist  # 1 = touching, 0 = far
                salience = salience_weights[typ] * proximity
                
                if salience > max_salience:
                    max_salience = salience
                
                if typ == 3 and salience > max_target_salience:
                    max_target_salience = salience
                    salient_target_ray = ray

        # Vector pointing toward most salient target
        vis_vec = np.array([0.0, 0.0])
        if salient_target_ray:
            angle = salient_target_ray.get("angle", 0.0)
            vis_vec = np.array([np.cos(angle), np.sin(angle)])

        # Novelty = change in salience (visual surprise)
        raw_novelty = abs(max_salience - self.ema_salience)
        self.ema_salience = 0.9 * self.ema_salience + 0.1 * max_salience

        # Feature vector [wall_proximity, pred_proximity, targ_proximity, novelty]
        features = np.array([
            _clamp(1.0 - wall_dist / 15.0, 0.0, 1.0),
            _clamp(1.0 - predator_dist / 15.0, 0.0, 1.0),
            _clamp(1.0 - target_dist / 15.0, 0.0, 1.0),
            _clamp(raw_novelty, 0.0, 1.0)
        ])

        return {
            "sensory.vision_vec": vis_vec,
            "sensory.vision_features": features,
            "sensory.novelty": float(raw_novelty)
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "ema_salience": float(self.ema_salience)
        }
