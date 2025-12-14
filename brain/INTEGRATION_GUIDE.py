"""
INTEGRATION GUIDE: Adding Model Selection to app.py

This file shows the changes needed to support switching between
the full brain and baseline models via the GUI.

=== STEP 1: Add Import ===

Near the top of app.py (around line 26), add:

    from brain.baseline_models import create_model, list_models, RatModel

=== STEP 2: Add Model to SimulationState.__init__ ===

In the SimulationState class __init__ (around line 400), add:

    # Active model selection (None = use full brain)
    self.active_baseline_model: Optional[RatModel] = None
    self.config["rat_model"] = "full_brain"  # Default

=== STEP 3: Modify process_votes to use active model ===

In process_votes(), near the start (after building sensory data), add a branch:

    # === BASELINE MODEL OVERRIDE ===
    if self.active_baseline_model is not None:
        # Build sensory dict for baseline model
        baseline_sensory = {
            "vision_buffer": vision_data,
            "whisker_hits": whisker_hits,
            "collision": collision,
            "danger": float(danger_level),
            "rat_pos": rat_pos_arr.copy(),
            "target_pos": sim_target_pos.copy() if sim_target_pos is not None else None,
        }
        
        # Get heading from baseline model
        heading = self.active_baseline_model.step(dt, baseline_sensory, reward_signal)
        
        # Return early with minimal processing
        # (You'll need to adjust the return signature or create dummy values)
        angle = float(np.arctan2(heading[1], heading[0]))
        if angle < 0:
            angle += 2 * np.pi
        
        # Create dummy return values for compatibility
        # ... (see STEP 3b below for full implementation)
        
        return (dummy_values...)

=== STEP 3b: Full baseline model branch ===

Here's a more complete version of the baseline branch. Add this near the 
START of process_votes(), after the initial sensory gathering but before
the full brain processing:

```python
# === BASELINE MODEL MODE ===
if self.active_baseline_model is not None:
    # Build sensory input for baseline model
    baseline_sensory = {
        "vision_buffer": vision_data,
        "whisker_hits": whisker_hits,
        "collision": collision,
        "danger": float(danger_level),
        "rat_pos": rat_pos_arr.copy(),
        "target_pos": sim_target_pos.copy() if sim_target_pos is not None else None,
    }
    
    # Step the baseline model
    heading_vec = self.active_baseline_model.step(dt, baseline_sensory, reward_signal)
    
    # Compute angle for return
    angle = float(np.arctan2(heading_vec[1], heading_vec[0]))
    if angle < 0:
        angle += 2 * np.pi
    
    # Minimal pain/touch from collisions
    pain = 1.0 if collision else 0.0
    touch = 1.0 if whisker_hits and any(whisker_hits) else 0.0
    
    # Return dummy values for all the brain metrics
    # These match the expected return signature of process_votes
    dummy_density = np.zeros((32, 13))  # soma visualization
    dummy_theta = np.zeros((32, 13))
    dummy_grid = self.grid_phases.copy() if hasattr(self, 'grid_phases') else np.zeros(3)
    dummy_soma_r = {"coherence": 0.0, "entropy": 0.0, "contrast": 0.0, "collapse_ema": 0.0}
    dummy_theta_r = {"coherence": 0.0, "entropy": 0.0, "contrast": 0.0, "collapse_ema": 0.0}
    dummy_place_metrics = {"place_nav_mag": 0.0, "place_act_max": 0.0, "place_goal_strength": 0.0}
    
    glycogen = 1.0
    atp = 1.0
    mt_plasticity = 1.0
    mt_gate = 0.5
    dopamine = 0.0
    dopamine_phasic = 0.0
    
    return (
        dummy_density,      # soma_density
        dummy_theta,        # d_theta
        dummy_grid,         # d_grid
        angle,              # final_decision (heading angle)
        glycogen,           # glycogen_level
        atp,                # atp_level
        pain,               # pain
        touch,              # touch
        dummy_place_metrics,# place_metrics
        dummy_soma_r,       # soma_r
        dummy_theta_r,      # theta_r
        mt_plasticity,      # mt_plasticity_mult
        mt_gate,            # mt_gate_thr
        dopamine,           # internal_dopamine
        dopamine_phasic,    # internal_dopamine_phasic
    )
```

=== STEP 4: Add model switching to /config endpoint ===

In the config() route handler (around line 4046), add handling for model changes:

```python
# === RAT MODEL SELECTION ===
if "rat_model" in payload:
    new_model = str(payload["rat_model"])
    old_model = sim.config.get("rat_model", "full_brain")
    
    if new_model != old_model:
        sim.config["rat_model"] = new_model
        
        if new_model == "full_brain":
            sim.brain.active_baseline_model = None
            print(f"MODEL: Switched to Full Brain")
        else:
            from brain.baseline_models import create_model
            sim.brain.active_baseline_model = create_model(
                new_model, 
                sim.rng_brain, 
                sim.config
            )
            print(f"MODEL: Switched to {new_model}")
```

=== STEP 5: Add model list endpoint ===

Add a new route to expose available models to the frontend:

```python
@app.route("/models/list", methods=["GET"])
def list_available_models():
    from brain.baseline_models import list_models
    return jsonify({"models": list_models()})
```

=== STEP 6: Expose current model in state response ===

In the step() route, add the current model to the response:

```python
# In the state dict being returned:
"rat_model": sim.config.get("rat_model", "full_brain"),
"baseline_diagnostics": sim.brain.active_baseline_model.get_diagnostics() 
    if sim.brain.active_baseline_model else {},
```

=== STEP 7: Frontend Changes (index.html) ===

Add a model selector dropdown to your control panel:

```html
<div class="control-group">
    <label for="rat-model">Rat Model:</label>
    <select id="rat-model" onchange="updateConfig()">
        <option value="full_brain">Full Brain (Neuromodulated)</option>
        <option value="random_walk">Random Walk</option>
        <option value="braitenberg">Braitenberg (Reactive)</option>
        <option value="braitenberg_drive">Braitenberg + Drive</option>
        <option value="q_learning">Q-Learning (Tabular RL)</option>
    </select>
</div>
```

And in your updateConfig() JavaScript function:

```javascript
function updateConfig() {
    const config = {
        // ... existing config fields ...
        rat_model: document.getElementById('rat-model').value,
    };
    
    fetch('/config', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(config)
    });
}
```

=== TESTING ===

Once integrated, you can:

1. Run the simulation with "Full Brain" selected
2. Switch to "Random Walk" - rat should move erratically
3. Switch to "Braitenberg" - rat should react to stimuli but have no memory
4. Switch to "Q-Learning" - rat should gradually learn
5. Compare performance metrics across models

Key metrics to track:
- Time to reach target (latency)
- Distance traveled
- Number of collisions
- Success rate in T-maze
- Thigmotaxis ratio in open field
"""

# This file is documentation only - no executable code
