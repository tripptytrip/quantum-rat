from abc import ABC, abstractmethod
from typing import Dict, Any
from .types import NeuromodulatoryState, SensoryState, CognitiveState, LearningSignals

class BrainComponent(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def step(self, dt: float, neuromod: NeuromodulatoryState, sensory: SensoryState, cognitive: CognitiveState, learning: LearningSignals) -> Dict[str, Any]:
        """
        Execute logic. Return a dictionary of updates to buses.
        Keys should be format: 'bus_name.field_name' (e.g., 'neuromod.norepinephrine_tonic')
        """
        pass

    @abstractmethod
    def reset(self):
        pass

    def get_diagnostics(self) -> Dict[str, Any]:
        return {}
