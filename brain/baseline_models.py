"""
Baseline Rat Models for Comparative Testing

These models provide null hypotheses for behavioral experiments.
Each implements the same interface so they can be swapped into
the simulation for direct comparison.

Usage:
    model = create_model("braitenberg", rng, config)
    heading = model.step(dt, sensory_data, reward)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np


class RatModel(ABC):
    """
    Abstract interface for all rat models.
    
    All models receive the same sensory input and produce a heading vector.
    This allows direct behavioral comparison across architectures.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name for UI display."""
        pass
    
    @abstractmethod
    def step(self, dt: float, sensory: Dict[str, Any], reward: float) -> np.ndarray:
        """
        Process one timestep.
        
        Args:
            dt: Time delta in seconds
            sensory: Dictionary containing:
                - vision_buffer: List of {dist, type, angle} ray hits
                - whisker_hits: List of bool for whisker contacts
                - collision: bool for wall collision
                - danger: float 0-1 predator proximity
                - rat_pos: np.array [x, y] current position
                - target_pos: np.array [x, y] goal position (if visible)
            reward: Reward signal (0 or positive)
            
        Returns:
            heading_vec: np.array [x, y] normalized direction to move
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset internal state for new episode."""
        pass
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Return internal state for debugging/visualization."""
        return {}


class RandomWalkModel(RatModel):
    """
    Pure random walk baseline.
    
    Changes direction randomly, with some persistence.
    Should be beaten by any sensible agent.
    """
    
    @property
    def name(self) -> str:
        return "Random Walk"
    
    def __init__(self, rng: np.random.Generator, config: Dict[str, Any]):
        self.rng = rng
        self.config = config
        self.current_heading = self.rng.random() * 2 * np.pi
        self.turn_probability = config.get("random_walk_turn_prob", 0.05)
        self.turn_magnitude = config.get("random_walk_turn_mag", 0.5)
    
    def reset(self):
        self.current_heading = self.rng.random() * 2 * np.pi
    
    def step(self, dt: float, sensory: Dict[str, Any], reward: float) -> np.ndarray:
        # Random direction changes
        if self.rng.random() < self.turn_probability:
            self.current_heading += self.rng.normal(0, self.turn_magnitude)
        
        # Bounce off walls (if collision, reverse)
        if sensory.get("collision", False):
            self.current_heading += np.pi + self.rng.normal(0, 0.3)
        
        # Normalize heading
        self.current_heading = self.current_heading % (2 * np.pi)
        
        return np.array([np.cos(self.current_heading), np.sin(self.current_heading)])
    
    def get_diagnostics(self) -> Dict[str, Any]:
        return {"heading": self.current_heading}


class BraitenbergModel(RatModel):
    """
    Braitenberg vehicle - pure stimulus-response.
    
    - Attracted to targets (food)
    - Repelled by predators
    - Avoids walls
    
    No memory, no learning, no internal state beyond current heading.
    Tests whether reactive behavior alone is sufficient.
    """
    
    @property
    def name(self) -> str:
        return "Braitenberg"
    
    def __init__(self, rng: np.random.Generator, config: Dict[str, Any]):
        self.rng = rng
        self.config = config
        self.current_heading = 0.0
        
        # Weights for different stimuli
        self.target_weight = config.get("braitenberg_target_weight", 2.0)
        self.predator_weight = config.get("braitenberg_predator_weight", -3.0)
        self.wall_weight = config.get("braitenberg_wall_weight", -1.0)
        self.noise_scale = config.get("braitenberg_noise", 0.1)
    
    def reset(self):
        self.current_heading = self.rng.random() * 2 * np.pi
    
    def step(self, dt: float, sensory: Dict[str, Any], reward: float) -> np.ndarray:
        vision = sensory.get("vision_buffer", [])
        
        # Accumulate weighted direction vectors
        heading_vec = np.zeros(2)
        
        for ray in vision:
            dist = ray.get("dist", 15.0)
            typ = ray.get("type", 0)
            angle = ray.get("angle", 0.0)
            
            # Proximity-based strength (closer = stronger)
            proximity = max(0, 1.0 - dist / 15.0)
            direction = np.array([np.cos(angle), np.sin(angle)])
            
            if typ == 1:  # Wall
                heading_vec += self.wall_weight * proximity * direction
            elif typ == 2:  # Predator
                heading_vec += self.predator_weight * proximity * direction
            elif typ == 3:  # Target
                heading_vec += self.target_weight * proximity * direction
        
        # Add noise
        heading_vec += self.rng.normal(0, self.noise_scale, size=2)
        
        # Handle collision - strong reversal
        if sensory.get("collision", False):
            heading_vec = -heading_vec + self.rng.normal(0, 0.5, size=2)
        
        # Normalize
        mag = np.linalg.norm(heading_vec)
        if mag > 0.01:
            heading_vec = heading_vec / mag
        else:
            # No stimuli - continue current heading with drift
            self.current_heading += self.rng.normal(0, 0.1)
            heading_vec = np.array([np.cos(self.current_heading), np.sin(self.current_heading)])
        
        # Update internal heading for continuity
        self.current_heading = np.arctan2(heading_vec[1], heading_vec[0])
        
        return heading_vec
    
    def get_diagnostics(self) -> Dict[str, Any]:
        return {"heading": self.current_heading}


class DriveBraitenbergModel(RatModel):
    """
    Braitenberg + Drive signal.
    
    Same as Braitenberg but with:
    - Hunger/restlessness that increases over time without reward
    - Increased exploration when drive is high
    
    Tests whether drive alone (without full neuromodulation) helps.
    """
    
    @property
    def name(self) -> str:
        return "Braitenberg + Drive"
    
    def __init__(self, rng: np.random.Generator, config: Dict[str, Any]):
        self.rng = rng
        self.config = config
        self.current_heading = 0.0
        
        # Braitenberg weights
        self.target_weight = config.get("braitenberg_target_weight", 2.0)
        self.predator_weight = config.get("braitenberg_predator_weight", -3.0)
        self.wall_weight = config.get("braitenberg_wall_weight", -1.0)
        self.base_noise = config.get("braitenberg_noise", 0.1)
        
        # Drive parameters
        self.drive_tau = config.get("drive_tau", 30.0)
        self.time_since_reward = 0.0
        self.drive = 0.0
    
    def reset(self):
        self.current_heading = self.rng.random() * 2 * np.pi
        self.time_since_reward = 0.0
        self.drive = 0.0
    
    def step(self, dt: float, sensory: Dict[str, Any], reward: float) -> np.ndarray:
        # Update drive
        if reward > 0.1:
            self.time_since_reward = 0.0
        else:
            self.time_since_reward += dt
        
        self.drive = 1.0 - np.exp(-self.time_since_reward / self.drive_tau)
        
        vision = sensory.get("vision_buffer", [])
        
        # Accumulate weighted direction vectors
        heading_vec = np.zeros(2)
        
        for ray in vision:
            dist = ray.get("dist", 15.0)
            typ = ray.get("type", 0)
            angle = ray.get("angle", 0.0)
            
            proximity = max(0, 1.0 - dist / 15.0)
            direction = np.array([np.cos(angle), np.sin(angle)])
            
            if typ == 1:  # Wall
                heading_vec += self.wall_weight * proximity * direction
            elif typ == 2:  # Predator
                heading_vec += self.predator_weight * proximity * direction
            elif typ == 3:  # Target
                # Drive increases target attraction
                weight = self.target_weight * (1.0 + self.drive)
                heading_vec += weight * proximity * direction
        
        # Noise increases with drive (restlessness)
        noise_scale = self.base_noise * (1.0 + 2.0 * self.drive)
        heading_vec += self.rng.normal(0, noise_scale, size=2)
        
        # Handle collision
        if sensory.get("collision", False):
            heading_vec = -heading_vec + self.rng.normal(0, 0.5, size=2)
        
        # Normalize
        mag = np.linalg.norm(heading_vec)
        if mag > 0.01:
            heading_vec = heading_vec / mag
        else:
            # High drive = more random exploration
            turn_rate = 0.1 + 0.4 * self.drive
            self.current_heading += self.rng.normal(0, turn_rate)
            heading_vec = np.array([np.cos(self.current_heading), np.sin(self.current_heading)])
        
        self.current_heading = np.arctan2(heading_vec[1], heading_vec[0])
        
        return heading_vec
    
    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "heading": self.current_heading,
            "drive": self.drive,
            "time_since_reward": self.time_since_reward
        }


class QLearningModel(RatModel):
    """
    Tabular Q-Learning baseline.
    
    Discretizes space into a grid and learns Q-values for each state-action pair.
    Standard RL baseline without any neuromodulatory complexity.
    
    Tests whether plain RL matches your architecture's performance.
    """
    
    @property
    def name(self) -> str:
        return "Q-Learning"
    
    def __init__(self, rng: np.random.Generator, config: Dict[str, Any]):
        self.rng = rng
        self.config = config
        
        # Grid discretization
        self.grid_size = config.get("q_grid_size", 20)
        self.n_actions = 8  # 8 directions
        
        # Q-table: [x, y, action] -> value
        self.q_table = np.zeros((self.grid_size, self.grid_size, self.n_actions))
        
        # Learning parameters
        self.alpha = config.get("q_alpha", 0.1)  # Learning rate
        self.gamma = config.get("q_gamma", 0.95)  # Discount factor
        self.epsilon = config.get("q_epsilon", 0.2)  # Exploration rate
        self.epsilon_decay = config.get("q_epsilon_decay", 0.999)
        self.min_epsilon = config.get("q_min_epsilon", 0.05)
        
        # State tracking
        self.last_state = None
        self.last_action = None
        self.current_heading = 0.0
        
        # Map size (will be updated from sensory)
        self.map_size = config.get("map_size", 30)
    
    def reset(self):
        self.last_state = None
        self.last_action = None
        self.current_heading = self.rng.random() * 2 * np.pi
        # Don't reset Q-table - learning persists across episodes
    
    def _pos_to_state(self, pos: np.ndarray) -> Tuple[int, int]:
        """Convert continuous position to discrete grid state."""
        x = int(np.clip(pos[0] / self.map_size * self.grid_size, 0, self.grid_size - 1))
        y = int(np.clip(pos[1] / self.map_size * self.grid_size, 0, self.grid_size - 1))
        return (x, y)
    
    def _action_to_heading(self, action: int) -> np.ndarray:
        """Convert discrete action to heading vector."""
        angle = action * (2 * np.pi / self.n_actions)
        return np.array([np.cos(angle), np.sin(angle)])
    
    def step(self, dt: float, sensory: Dict[str, Any], reward: float) -> np.ndarray:
        # Get current position
        pos = sensory.get("rat_pos", np.array([15.0, 15.0]))
        state = self._pos_to_state(pos)
        
        # Q-learning update (if we have a previous state)
        if self.last_state is not None and self.last_action is not None:
            old_q = self.q_table[self.last_state[0], self.last_state[1], self.last_action]
            max_next_q = np.max(self.q_table[state[0], state[1], :])
            
            # Penalty for collision
            if sensory.get("collision", False):
                reward -= 0.5
            
            # Q-update
            new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
            self.q_table[self.last_state[0], self.last_state[1], self.last_action] = new_q
        
        # Epsilon-greedy action selection
        if self.rng.random() < self.epsilon:
            action = self.rng.integers(0, self.n_actions)
        else:
            action = np.argmax(self.q_table[state[0], state[1], :])
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        # Store for next update
        self.last_state = state
        self.last_action = action
        
        # Convert action to heading
        heading_vec = self._action_to_heading(action)
        
        # Handle collision - override with reversal
        if sensory.get("collision", False):
            heading_vec = -heading_vec
            heading_vec += self.rng.normal(0, 0.3, size=2)
            mag = np.linalg.norm(heading_vec)
            if mag > 0.01:
                heading_vec = heading_vec / mag
        
        self.current_heading = np.arctan2(heading_vec[1], heading_vec[0])
        
        return heading_vec
    
    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "heading": self.current_heading,
            "epsilon": self.epsilon,
            "q_mean": float(np.mean(self.q_table)),
            "q_max": float(np.max(self.q_table)),
            "last_state": self.last_state
        }


# === Model Registry ===

MODEL_REGISTRY: Dict[str, type] = {
    "random_walk": RandomWalkModel,
    "braitenberg": BraitenbergModel,
    "braitenberg_drive": DriveBraitenbergModel,
    "q_learning": QLearningModel,
}


def create_model(model_type: str, rng: np.random.Generator, config: Dict[str, Any]) -> RatModel:
    """
    Factory function to create a rat model by type name.
    
    Args:
        model_type: One of 'random_walk', 'braitenberg', 'braitenberg_drive', 'q_learning'
        rng: NumPy random generator for reproducibility
        config: Configuration dictionary
        
    Returns:
        RatModel instance
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_type](rng, config)


def list_models() -> Dict[str, str]:
    """Return dict of model_type -> display_name for UI."""
    return {
        "full_brain": "Full Brain (Neuromodulated)",
        "random_walk": "Random Walk",
        "braitenberg": "Braitenberg (Reactive)",
        "braitenberg_drive": "Braitenberg + Drive",
        "q_learning": "Q-Learning (Tabular RL)",
    }
