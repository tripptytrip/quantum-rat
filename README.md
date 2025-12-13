# Quantum Rat

A computational neuroethology platform exploring the minimal sufficient architecture for lifelike adaptive behavior.

## What This Is

This project simulates a virtual rat navigating a maze environment, driven by interacting brain subsystems rather than hand-coded behaviors. The goal isn't to replicate biological mechanisms exactly, but to test whether specific architectural patterns—when coupled together—produce mammalian-typical behavioral signatures.

The central question: **What's the minimum set of interacting systems needed to generate adaptive, lifelike behavior?**

## Philosophical Position

This project is built on a few explicit assumptions:

- **Hard determinism**: Behavior emerges from causal mechanisms. No free will is posited or required.
- **Consciousness is not the target**: We're not trying to create or prove consciousness. We're interested in whether the *functional relationships* between subsystems produce recognizable behavioral outputs.
- **Quantum framing is metaphorical**: The "microtubule" and "coherence/collapse" language in the code is a computational abstraction, not a literal claim about quantum effects in neurons. It provides a useful dynamics for state-dependent gating and exploration/exploitation tradeoffs.

## Architecture

The rat brain consists of modular, interacting subsystems:

### Microtubule Fields (Soma + Theta)
Coupled oscillator fields with coherence/entropy dynamics. High coherence biases toward exploitation and gating stability. High entropy biases toward exploration and plasticity. A threshold-triggered "collapse" resets or binarizes state—functionally similar to attractor dynamics, not quantum mechanics.

### Predictive Processing
Learns to predict sensory states from internal context. Prediction error serves as a training signal and modulates exploration via the norepinephrine system.

### Neuromodulatory Systems
| System | Function |
|--------|----------|
| **Dopamine (PVLV)** | Reward prediction, phasic learning signals, motivation |
| **Serotonin** | Patience, impulse control, give-up threshold under stress |
| **Norepinephrine (LC)** | Arousal, uncertainty tracking, explore/exploit balance |
| **Acetylcholine (Basal Forebrain)** | Precision weighting, encoding vs recall modes |

### Thalamic Gating (TRN)
Switches between sensory-dominant and memory-dominant processing based on arousal and task demands. Can enter "microsleep" states when both channels burst, triggering offline consolidation.

### Spatial System
- **Place cells**: Location-specific firing, goal vector learning
- **Grid cells**: Path integration via periodic spatial coding
- **Head direction cells**: Compass-like orientation tracking

### Working Memory
Multiple slots with strength-based decay and competitive write access. Gated by novelty, salience, and dopamine signals.

### Hippocampal Replay
During microsleep, recent experiences are replayed and consolidated to basal ganglia weights. Supports offline learning and memory bridging across temporally distant events.

### Metabolism (Astrocyte Model)
ATP and glycogen dynamics create genuine resource constraints. Cognitive effort costs energy. Depleted states impair attention and decision-making.

## Behavioral Validation

The platform includes established behavioral assays with known mammalian baselines:

| Test | What It Measures |
|------|------------------|
| **Open Field** | Anxiety, exploration (thigmotaxis vs center time) |
| **T-Maze** | Working memory (delayed alternation) |
| **Morris Water Maze** | Spatial learning (latency reduction across trials) |
| **Survival Arena** | Threat response, predator evasion |

Success criterion: the simulated rat should produce behavioral patterns qualitatively similar to biological rats on these tasks—not through parameter tuning per task, but as emergent properties of the architecture.

## Research Program

### Current Focus
- Modular architecture allowing component swap-in for ablation studies
- Automated parameter sweeps with data recording
- Expanding the behavioral test battery

### Planned Ablation Comparisons
- Microtubule field → simple matched-statistics noise
- PVLV dopamine → fixed reward signal
- TRN gating → always-on (no attentional switching)
- Place cells → pure path integration
- Replay consolidation → no offline learning

Each comparison tests whether the added complexity produces measurable behavioral differences.

### Planned Additional Tests
- Elevated plus maze (anxiety with explicit safe/risky arms)
- Reversal learning (flexibility vs perseveration)
- Novel object recognition (familiarity detection)
- Learned helplessness paradigms (serotonin/give-up dynamics)

## Running the Simulation

### Requirements
```
flask
numpy
```

### Start
```bash
python app.py
```

Then open `http://localhost:5000` in a browser.

### Configuration
The simulation exposes many parameters via the `/config` endpoint, including:
- Neuromodulator gains
- Microtubule causal coupling strength
- Replay and consolidation settings
- Working memory parameters
- Predictive processing learning rates

A deterministic mode (fixed seed) is available for reproducible runs.

## What This Is Not

- **Not a claim about how brains actually work**: The architecture is inspired by neuroscience but uses computational stand-ins, not biologically realistic models.
- **Not trying to create consciousness**: The rat may "look alive" but we make no claims about subjective experience.
- **Not a finished product**: This is an ongoing exploration. Expect rough edges and evolving design.

## Contributing

Currently a solo project. If you're interested in computational neuroethology, ablation methodology, or just want to critique the architecture, feel free to open an issue.

## License

MIT

## Acknowledgments

Inspired by Braitenberg's *Vehicles*, predictive processing frameworks (Friston, Clark), and the broader project of understanding behavior as emergent from mechanism.