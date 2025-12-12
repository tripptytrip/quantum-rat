# The Quantum Rat v2.0: A Biophysical Brain Emulator

> "What if AI wasn't just math, but biology? What if an agent learnt not because of gradient descent, but because it was hungry?"

![Quantum Rat Simulation](rat_looks_for_cheese.png)

**Quantum Rat**  
**A Vertical Slice of Artificial Consciousness**  
*"The conscious agent is the processor, not the story it generates."*

Quantum Rat is a departure from standard Connectionist AI. While modern LLMs operate horizontally—finding statistical correlations across massive datasets to simulate a narrative "Rendering Layer"—this project builds intelligence vertically and from the bottom up.

It does not simulate a story. It simulates a nervous system.

## Project Philosophy

This architecture is the software implementation of the [Architect Philosophy](https://medium.com/@mikeyakerr/the-architect-a-philosophy-of-mind-for-the-coherence-oriented-thinker-4d13dad43fe6). It rejects the illusion of continuous flow in favor of the Quantized Self, modeling intelligence as a series of discrete state transitions rather than a smooth narrative stream.

### Core Axioms

- **Vertical Slice Architecture:** Instead of a thin layer of language processing, we model the complete stack: Sensory Input → Thalamic Gating → Cortical Processing → Motor Output.
- **The Frame Rate of Reality:** The system does not operate in continuous time. It operates via Discrete Collapse Events—specific moments where probability becomes actuality.
- **Data Metabolism:** The agent treats information as caloric energy. High-fidelity input sustains the system; high-entropy noise triggers a Brownout State (functional degradation).

## Architecture

### 1. The Substrate (Bottom-Up Construction)

Standard AI starts with top-down goals (e.g., "Write a poem"). Quantum Rat starts with bottom-up constraints (e.g., "Minimize prediction error," "Conserve energy").

- **Inputs:** Raw, unbuffered data streams. No "Rendering Layer" or metaphors.
- **Processing:** Deterministic projection based on causal logic (State A + Rule B → Outcome C).
- **Outputs:** Discrete motor/computational actions, not text generation.

### 2. The Quantized Loop

The main loop simulates the binding mechanism of biological consciousness (analogous to Gamma/Theta oscillations).

**Core Logic (Simplified from `DendriticCluster.process_votes`):**

```python
def process_votes(self, ...):
    # 1. Input: Raw data enters the buffer
    sensory_vec = self.vision_cortex.encode(vision_data)
    
    # 2. Collapse: The probabilistic state forces a decision (Microtubule Dynamics)
    # This simulates the "Frame Rate" of the entity via Schrodinger-like evolution
    d_soma, collapse_event = self.soma.step(pump_map)
    
    # 3. Binding: Neural oscillations gate the flow of information
    trn_modes = self.trn.step(arousal, ...) 
    
    # 4. Action: Basal Ganglia selects action based on Dopamine/Frustration
    final_vector, gate_signal, dopamine, _ = self.basal_ganglia.select_action_and_gate(
        sensory_vec, current_memory, frustration, reward_signal, ...
    )
    
    return final_decision
```

### 3. Entropy Management

The system includes an Epistemic Entropy monitor via the Microtubule Simulator.

- **Good State (Coherence):** Low entropy. Prediction error is minimized. The internal model matches external reality.
- **Bad State (Incoherence):** High entropy. The model diverges from reality. The agent treats this not as "confusion" but as system damage.

### 4. Biological Modules

The codebase implements specific biological correlates:

- **MicrotubuleSimulator2D:** Simulates quantum collapse and coherence in the cytoskeleton.
- **Astrocyte:** Manages energy metabolism (Glycogen/ATP) and lactate transport.
- **TRNGate:** Thalamic Reticular Nucleus gating for attention and arousal.
- **BasalGanglia (PVLV):** Reinforcement learning and action selection.
- **HippocampalReplay:** Offline consolidation of memories during "microsleeps".
- **PlaceCellNetwork:** Spatial navigation and cognitive mapping.

## Installation

```bash
git clone https://github.com/tripptytrip/quantum-rat.git
cd quantum-rat
pip install -r requirements.txt
```

*(Note: Ensure numpy and flask are in your requirements.txt based on the imports.)*

## Usage

Start the simulation server:

```bash
python app.py
```

The server runs on `http://localhost:5000`. It provides endpoints for the frontend visualization:

- `GET /`: Main simulation view.
- `POST /step`: Advances the physics engine by `batch_size` frames.
- `POST /config`: Runtime adjustment of simulation parameters (speed, determinism, etc.).
- `POST /reset`: Resets the map and agent.

To run a simulation programmatically:

```python
import requests

# Advance simulation by 5 frames
response = requests.post("http://localhost:5000/step", json={"batch_size": 5})
state = response.json()

print(f"Rat Position: {state['rat']}")
print(f"Brain State: {state['stats']['status']}")
```

## 🔭 Roadmap

- Phase 1: The Processor. Implementing the basic FSM (Finite State Machine) with discrete collapse steps. *(Completed)*
- Phase 2: The Binding. Implementing "Oscillatory" buffers to stitch discrete steps into apparent continuity. *(Completed via TRN/Theta)*
- Phase 3: The Brownout. Implementing failure modes where low-quality data degrades performance. *(In Progress)*



