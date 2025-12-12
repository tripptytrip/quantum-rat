# The Quantum Rat v3.5: A Bio-Plausible Cognitive Architecture

> "What if AI wasn't just math, but biology? What if an agent learnt not because of gradient descent, but because it was hungry, confused, or tired?"

![Quantum Rat Simulation](rat_looks_for_cheese.png)

**Quantum Rat**  
**A Vertical Slice of Artificial Consciousness**  
*"The conscious agent is the processor, not the story it generates."*

Quantum Rat is a departure from standard Connectionist AI. While modern LLMs operate horizontally—finding statistical correlations across massive datasets to simulate a narrative "Rendering Layer"—this project builds intelligence vertically and from the bottom up.

It does not simulate a story. It simulates a nervous system.

---

## v3.5 Major Features

- **🧪 Scientific Test Bed:** A modular laboratory protocol system allowing for rigorous cognitive testing (Open Field, T-Maze, Water Maze, Survival Arena).
- **🔋 Metabolic Cost of Thought:** Thinking is expensive. The **Astrocyte System** tracks ATP and Glycogen. High cortical load (planning) drains energy; low energy triggers "Brain Fog," forcing the agent to revert to reflexive habits.
- **🌀 Entropy as Curiosity:** The **Microtubule Field** tracks "Epistemic Entropy." High confusion injects quantum noise into the Thalamus, driving exploration and "jitter" until coherence is restored.
- **🔦 Thalamic Searchlight:** Top-down attention from the Cortex can bias the Thalamus to suppress noise, but this requires sufficient ATP to maintain focus.
- **💤 Quantum Replay (Dreams):** The agent requires "Microsleeps" to consolidate memories. During these states, the Hippocampus pumps memory buffers into the Microtubule field to find coherent patterns, which are then hard-coded into the Basal Ganglia.

---

## Project Philosophy

This architecture is the software implementation of the [Architect Philosophy](https://medium.com/@mikeyakerr/the-architect-a-philosophy-of-mind-for-the-coherence-oriented-thinker-4d13dad43fe6). It rejects the illusion of continuous flow in favor of the Quantized Self, modeling intelligence as a series of discrete state transitions rather than a smooth narrative stream.

### Core Axioms

- **Vertical Slice Architecture:** Instead of a thin layer of language processing, we model the complete stack: Sensory Input → Thalamic Gating → Cortical Processing → Motor Output.
- **The Frame Rate of Reality:** The system does not operate in continuous time. It operates via Discrete Collapse Events—specific moments where probability becomes actuality.
- **Data Metabolism:** The agent treats information as caloric energy. High-fidelity input sustains the system; high-entropy noise triggers a Brownout State (functional degradation).

---

## Architecture

### 1. The Substrate (Bottom-Up Construction)

Standard AI starts with top-down goals (e.g., "Write a poem"). Quantum Rat starts with bottom-up constraints (e.g., "Minimize prediction error," "Conserve energy").

- **Inputs:** Raw, unbuffered data streams (Vision Rays, Whisker Hits). No "Rendering Layer" or metaphors.
- **Processing:** Deterministic projection based on causal logic (State A + Rule B → Outcome C).
- **Outputs:** Discrete motor actions, not text generation.

### 2. The Quantized Loop

The main loop simulates the binding mechanism of biological consciousness (analogous to Gamma/Theta oscillations).

**Core Logic (Simplified from `DendriticCluster.process_votes`):**

```python
def process_votes(self, ...):
    # 1. Metabolism: Check ATP. If low, throttle cortical gain ("Brain Fog").
    real_atp, ext_lactate = self.astrocyte.step(neural_activity, cortical_load)

    # 2. Entropy: Calculate confusion (Entropy - Coherence).
    # If confused, inject noise into Thalamus (Curiosity).
    confusion_index = clamp(epistemic_entropy - coherence, 0.0, 1.0)

    # 3. Input: Thalamus gates sensory data based on Attention & ATP.
    sensory_vec = self.thalamus.relay(vis_data, ..., noise_level=confusion_index)

    # 4. Action: Basal Ganglia selects action based on Dopamine & Replay history.
    final_vector, gate_signal, ... = self.basal_ganglia.select_action_and_gate(...)

    # 5. Collapse: Update Microtubule Field (Orch OR event).
    d_soma, _ = self.soma.step(pump_map)

    return final_decision
```

### 3. Biological Modules

The codebase implements specific biological correlates:

- **MicrotubuleSimulator2D:** Simulates quantum collapse, coherence, and epistemic entropy.
- **Astrocyte:** Manages energy metabolism (Glycogen/ATP), enforcing the "Cost of Thought."
- **TRNGate:** Thalamic Reticular Nucleus gating for attention and arousal states (Burst vs Tonic).
- **BasalGanglia (PVLV):** Reinforcement learning, action selection, and predictive intercept logic.
- **HippocampalReplay:** Offline consolidation of memories. Implements "Quantum Dreaming" where coherent patterns in the buffer are permanently stored.
- **PlaceCellNetwork:** Spatial navigation and cognitive mapping (Grid Cells / Head Direction Cells).

---

## Installation

```bash
git clone https://github.com/tripptytrip/quantum-rat.git
cd quantum-rat
pip install -r requirements.txt
```

*(Note: Requires `numpy`, `flask`.)*

---

## Usage

Start the simulation server:

```bash
python app.py
```

The server runs on `http://localhost:5000`.

---

## Frontend Controls (v3.5)

- **Monitor:** Real-time visualization of brain states (Soma, Theta, Grid Cells) and physiological metrics (Dopamine, ATP, Frustration).
- **Controls:** Adjust simulation speed, ablate brain regions (lobotomize the Hippocampus), or toggle metabolic constraints.
- **Research:** Access the **Scientific Test Bed**:
  - **Open Field:** Test anxiety and exploration.
  - **T-Maze:** Test working memory (delayed alternation).
  - **Water Maze:** Test long-term spatial memory consolidation.
  - **Survival Arena:** Test predictive evasion against an aggressive predator.

---

## Roadmap

- **Phase 1: The Processor.** Implementing the basic FSM with discrete collapse steps. (Completed)
- **Phase 2: The Binding.** Implementing "Oscillatory" buffers to stitch steps into continuity. (Completed via TRN/Theta)
- **Phase 3: The Organism.** Implementing metabolic constraints, sleep cycles, and entropy-driven curiosity. (Completed v3.5)
- **Phase 4: The Society.** Multi-agent interaction and social entropy. (Planned)