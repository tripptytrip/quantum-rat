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

    def step(self, dt: float, neuromod: NeuromodulatoryState, sensory: SensoryState, cognitive: CognitiveState, learning: LearningSignals) -> Dict[str, Any]:
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
        # --- BOREDOM CIRCUIT STATE ---
        self.last_vis_vec = np.zeros(2)
        self.boredom = 0.0

    def reset(self):
        self.ema_salience = 0.0
        self.last_vis_vec = np.zeros(2)
        self.boredom = 0.0

    def step(self, dt: float, neuromod: NeuromodulatoryState, sensory: SensoryState, cognitive: CognitiveState, learning: LearningSignals) -> Dict[str, Any]:
        vision_rays = sensory.raw_vision_buffer
        
        # Default Weights
        salience_weights = {1: 0.5, 2: 2.0, 3: 1.5} # wall, predator, target
        
        max_salience = 0.0
        max_target_salience = 0.0
        salient_target_ray = None

        wall_dist = 15.0
        predator_dist = 15.0
        target_dist = 15.0

        for ray in vision_rays:
            dist = ray.get("dist", 15.0)
            typ = ray.get("type", 0)
            
            # Distance calc
            if typ == 1: wall_dist = min(wall_dist, dist)
            if typ == 2: predator_dist = min(predator_dist, dist)
            if typ == 3: target_dist = min(target_dist, dist)

            # Salience calc
            if typ in salience_weights:
                salience = salience_weights[typ] * (1.0 / (dist + 1e-9))
                if salience > max_salience:
                    max_salience = salience
                
                if typ == 3 and salience > max_target_salience:
                    max_target_salience = salience
                    salient_target_ray = ray

        # Vector Calculation
        vis_vec = np.array([0.0, 0.0])
        if salient_target_ray:
            angle = salient_target_ray.get("angle", 0.0)
            vis_vec = np.array([np.cos(angle), np.sin(angle)])

        # Novelty Calc (Visual Change)
        raw_novelty = abs(max_salience - self.ema_salience)
        self.ema_salience = 0.9 * self.ema_salience + 0.1 * max_salience

        # --- BOREDOM CIRCUIT ---
        # 1. Detect Stagnation: How much has the visual field changed?
        # If the rat is sitting still, this distance will be near zero.
        visual_delta = np.linalg.norm(vis_vec - self.last_vis_vec)
        
        # 2. Integrate Boredom
        # If nothing changes for ~2-3 seconds, boredom peaks.
        if visual_delta < 0.02: 
            self.boredom += 0.3 * dt  # Rises
        else:
            self.boredom -= 2.0 * dt  # Drops instantly if scene changes
        
        self.boredom = _clamp(self.boredom, 0.0, 1.0)
        self.last_vis_vec = vis_vec.copy()

        # 3. Output Effective Novelty
        # If bored, we inject 'Novelty' to trick the LC into triggering Exploration
        effective_novelty = max(raw_novelty, self.boredom)

        # Feature Vector [wall, pred, targ, novelty]
        features = np.array([
            _clamp(1.0 - wall_dist / 15.0, 0.0, 1.0),
            _clamp(1.0 - predator_dist / 15.0, 0.0, 1.0),
            _clamp(1.0 - target_dist / 15.0, 0.0, 1.0),
            _clamp(effective_novelty, 0.0, 1.0)
        ])

        return {
            "sensory.vision_vec": vis_vec,
            "sensory.vision_features": features,
            "sensory.novelty": float(effective_novelty)
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "boredom": float(self.boredom),
            "visual_delta": float(np.linalg.norm(self.last_vis_vec)), # rough proxy for movement
            "novelty": float(self.ema_salience) # useful debug
        }
