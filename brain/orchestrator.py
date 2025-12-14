from typing import Dict, Any, List
import numpy as np
from .types import NeuromodulatoryState, SensoryState, MotorCommand, CognitiveState, LearningSignals
from .interface import BrainComponent

class BrainOrchestrator:
    def __init__(self, config: Dict[str, Any], rng):
        self.config = config
        self.rng = rng
        
        self.neuromod = NeuromodulatoryState()
        self.sensory = SensoryState()
        self.motor = MotorCommand()
        self.cognitive = CognitiveState()
        self.learning = LearningSignals()
        
        self.components: Dict[str, BrainComponent] = {}
        self.execution_order: List[str] = []
        self.last_diagnostics: Dict[str, Any] = {}

    def register(self, component: BrainComponent):
        self.components[component.name] = component
        if component.name not in self.execution_order:
            self.execution_order.append(component.name)

    def step(self, dt: float, raw_sensory: Dict[str, Any], reward: float):
        # 1. Ingest Raw Sensory Data
        # We now accept raw buffers. Legacy fields are kept as fallbacks if buffers are missing.
        self.sensory.raw_vision_buffer = raw_sensory.get("vision_buffer", [])
        self.sensory.raw_whisker_hits = raw_sensory.get("whisker_hits", [])
        self.sensory.raw_collision = raw_sensory.get("collision", False)

        # Shim Fallbacks (in case caller still passes pre-calculated values)
        if "novelty" in raw_sensory: self.sensory.novelty = raw_sensory.get("novelty", 0.0)
        if "pain" in raw_sensory: self.sensory.pain = raw_sensory.get("pain", 0.0)
        if "touch" in raw_sensory: self.sensory.touch = raw_sensory.get("touch", 0.0)
        
        # 2. Ingest Learning Signals
        self.learning.reward = reward
        self.learning.surprise = raw_sensory.get("surprise", 0.0)
        self.learning.danger = raw_sensory.get("danger", 0.0)

        # 3. Run Pipeline
        for name in self.execution_order:
            comp = self.components[name]
            outputs = comp.step(dt, self.neuromod, self.sensory, self.cognitive, self.learning)
            self._merge_outputs(outputs)

        # 4. Diagnostics
        for name, comp in self.components.items():
            self.last_diagnostics[name] = comp.get_diagnostics()

    def _merge_outputs(self, outputs: Dict[str, Any]):
        if not outputs: return
        for k, v in outputs.items():
            if "." not in k: continue
            domain, field = k.split('.', 1)
            
            target_bus = None
            if domain == 'neuromod': target_bus = self.neuromod
            elif domain == 'sensory': target_bus = self.sensory
            elif domain == 'motor': target_bus = self.motor
            elif domain == 'cognitive': target_bus = self.cognitive
            elif domain == 'learning': target_bus = self.learning
            
            if target_bus and hasattr(target_bus, field):
                setattr(target_bus, field, v)

    def reset(self):
        self.neuromod = NeuromodulatoryState()
        self.sensory = SensoryState()
        self.motor = MotorCommand()
        self.cognitive = CognitiveState()
        self.learning = LearningSignals()
        for c in self.components.values():
            c.reset()
