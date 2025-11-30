from flask import Flask, render_template, jsonify, request
import numpy as np
import json
import os
import copy
import csv

app = Flask(__name__)

# --- PHYSICS ENGINE ---
COMPLEX_TYPE = np.complex128

class MicrotubuleSimulator2D:
    def __init__(self, label, length_points=32, seam_shift=3, dt=0.01):
        self.label = label
        # In this model: X=13 (Circumference/Protofilaments), Y=Length
        self.Nx = 13  
        self.Ny = length_points
        self.dt = dt
        self.seam_shift = seam_shift # The "3-start helix" shift

        # Physics Constants
        self.H_BAR = 1.0     # Normalized Planck constant
        self.base_gamma = 1.5 # Dipole coupling strength
        
        # State Arrays
        self.psi = np.zeros((self.Ny, self.Nx), dtype=COMPLEX_TYPE)
        self.superradiance = np.zeros((self.Ny, self.Nx), dtype=float) # S_R parameter
        self.accumulated_Eg = 0.0
        
        # Phases: "ISOLATION" (Classical), "SUPERPOSITION" (Quantum), "COLLAPSE"
        self.phase = "ISOLATION"
        
        self.reset_wavefunction()

    def reset_wavefunction(self):
        # Initialize with random thermal noise
        noise_r = 0.1 * (np.random.rand(self.Ny, self.Nx) - 0.5)
        noise_i = 0.1 * (np.random.rand(self.Ny, self.Nx) - 0.5)
        self.psi = (noise_r + 1j * noise_i).astype(COMPLEX_TYPE)
        self.superradiance.fill(0.0)
        self.accumulated_Eg = 0.0
        self.phase = "ISOLATION"

    def get_helical_neighbor(self, grid, dx, dy):
        """
        Retrieves neighbor grid shifted by dx, dy.
        Handles Cylindrical Boundary Condition on X (Circumference) with Seam Shift.
        """
        # 1. Shift Y (Length) - No wrapping (Open ends)
        # We handle open ends by padding or masking later, for now we roll.
        shifted = np.roll(grid, dy, axis=0)
        
        # 2. Shift X (Circumference) - HELICAL WRAPPING
        # If we shift X, we must handle the seam.
        
        if dx == 0:
            return shifted
            
        # Standard roll for the bulk
        temp_grid = np.roll(shifted, dx, axis=1)
        
        # CORRECT THE SEAM:
        # If we rolled +1 (Right), the column at 0 came from -1. 
        # This wrapped column needs to be shifted by +seam_shift in Y.
        if dx == 1:
            # The column at index 0 is the one that wrapped. Shift it up.
            temp_grid[:, 0] = np.roll(temp_grid[:, 0], self.seam_shift)
        elif dx == -1:
            # The column at index -1 is the one that wrapped from 0. Shift it down.
            temp_grid[:, -1] = np.roll(temp_grid[:, -1], -self.seam_shift)
            
        return temp_grid

    def step(self, pump_map, lfp_signal=0.0, anesthetic_conc=0.0):
        # --- FIX: ANESTHESIA DAMPENS INPUT SENSITIVITY ---
        # High anesthesia blocks the 'pump' from exciting the tryptophan network
        effective_pump = pump_map * (1.0 - anesthetic_conc)
        
        # 1. Update Superradiance (Metabolic Pumping vs Thermal Decay)
        decay_rate = 0.1
        # Use effective_pump here
        self.superradiance = self.superradiance * (1 - decay_rate) + (effective_pump * 0.2)
        self.superradiance = np.clip(self.superradiance, 0, 1.0)

        # 2. Calculate Dipole Forces
        neighbors = [
            (1, 0, 1.0),   # Up (Y+)
            (-1, 0, 1.0),  # Down (Y-)
            (0, 1, 0.58),  # Right (X+)
            (0, -1, 0.58), # Left (X-)
            (1, 1, 0.4),   # Diag Up-Right
            (-1, -1, 0.4)  # Diag Down-Left
        ]
        
        dipole_potential = np.zeros_like(self.psi, dtype=COMPLEX_TYPE)
        
        for dy, dx, coupling in neighbors:
            psi_neighbor = self.get_helical_neighbor(self.psi, dx, dy)
            # Anesthesia ALSO dampens lateral coupling (existing logic)
            effective_coupling = coupling
            if dx != 0: 
                effective_coupling *= (1.0 - anesthetic_conc)
            
            dipole_potential += effective_coupling * psi_neighbor

        # 3. Phase Dynamics
        density = np.abs(self.psi)**2
        collapse_event = False
        
        avg_coherence = np.mean(self.superradiance)
        
        # If coherence is low (due to anesthesia blocking pump), we stay Classical
        if avg_coherence < 0.2:
            self.phase = "ISOLATION"
            d_psi = (np.real(dipole_potential) - 4*self.psi) * 0.1
            self.psi += d_psi
            self.psi /= (np.max(np.abs(self.psi)) + 1e-9) 
            
        else:
            if self.phase == "ISOLATION":
                self.phase = "SUPERPOSITION"
                self.accumulated_Eg = 0.0
            
            nonlinear = self.base_gamma * density * self.psi
            
            # --- FIX: PUMP DRIVES SCHRODINGER EVOLUTION ---
            # Add the pump term to the Hamiltonian evolution
            # i * dPsi/dt = H * Psi + Pump
            d_psi = -1j * (dipole_potential + nonlinear) + (effective_pump * self.psi)
            
            self.psi += d_psi * self.dt
            
            current_Eg = np.var(density) * 10.0
            self.accumulated_Eg += current_Eg * self.dt
            
            if self.accumulated_Eg > self.H_BAR:
                self.phase = "COLLAPSE"
                collapse_event = True
                if np.random.rand() > 0.5:
                     self.psi = np.where(density > np.mean(density), 1.0+0j, 0.0+0j)
                else:
                     self.reset_wavefunction()
                self.accumulated_Eg = 0.0
                self.phase = "ISOLATION" 

        norm = np.linalg.norm(self.psi)
        if norm > 0: self.psi /= norm
        
        viz_density = density * 200.0 
        
        return viz_density, collapse_event

# --- NEW: Astrocyte-Neuron Lactate Shuttle (ANLS) ---
class Astrocyte:
    def __init__(self):
        self.glycogen = 1.0       # The "Battery" (0.0 to 1.0)
        self.astro_lactate = 0.5  # Lactate inside Astrocyte
        self.ext_lactate = 0.5    # Lactate in Extracellular Space (ECS)
        self.neuron_lactate = 0.5 # Lactate inside Neuron
        self.neuronal_atp = 1.0   # Final ATP available for firing
        self.glutamate_load = 0.0 # Signal from synapse (Demand)
        
        # Kinetic Constants (Michaelis-Menten)
        self.Vmax_MCT4 = 0.8  # Astrocyte export capacity
        self.Km_MCT4 = 4.0    # Low affinity (needs high concentration to push out)
        
        self.Vmax_MCT2 = 0.6  # Neuron import capacity
        self.Km_MCT2 = 0.5    # High affinity (sucks in fuel even at low concentrations)
        
    def step(self, firing_rate, dt=0.05):
        # 1. Demand Signal (Glutamate uptake triggers glycolysis)
        demand = firing_rate * 0.1
        self.glutamate_load += demand
        
        # 2. Glycogenolysis (The Suzuki Effect)
        # We produce intracellular lactate in the astrocyte from glycogen reserves
        glycolysis_rate = 0.05 * self.glutamate_load * self.glycogen
        
        self.glycogen -= glycolysis_rate * dt
        self.astro_lactate += glycolysis_rate * dt
        
        # 3. Transport: Astrocyte -> ECS (via MCT4)
        # Equation: Rate = Vmax * [S] / (Km + [S])
        flux_astro_to_ecs = self.Vmax_MCT4 * (self.astro_lactate / (self.Km_MCT4 + self.astro_lactate))
        
        # 4. Transport: ECS -> Neuron (via MCT2)
        flux_ecs_to_neuron = self.Vmax_MCT2 * (self.ext_lactate / (self.Km_MCT2 + self.ext_lactate))
        
        # Update Compartments
        self.astro_lactate -= flux_astro_to_ecs * dt
        self.ext_lactate += (flux_astro_to_ecs - flux_ecs_to_neuron) * dt
        self.neuron_lactate += flux_ecs_to_neuron * dt
        
        # 5. Oxidative Phosphorylation (Neuron converts Lactate -> ATP)
        # This is the bottleneck: The neuron can only burn what MCT2 lets in.
        conversion_rate = 2.0 * self.neuron_lactate
        atp_production = conversion_rate * 3.0 # ATP yield efficiency
        
        self.neuron_lactate -= conversion_rate * dt
        
        # 6. Consumption (Basal + Activity)
        basal_metabolism = 0.005
        activity_cost = firing_rate * 0.05
        self.neuronal_atp += (atp_production - (basal_metabolism + activity_cost)) * dt
        
        # Decay signals
        self.glutamate_load *= 0.9
        
        # Clamping to biological limits
        self.glycogen = np.clip(self.glycogen, 0, 1.0)
        self.neuronal_atp = np.clip(self.neuronal_atp, 0, 1.5)
        
        # RETURN TWO VALUES: 
        # 1. ATP (Energy for motion/coherence)
        # 2. Extracellular Lactate (Signal for Plasticity/Learning)
        return self.neuronal_atp, self.ext_lactate

# --- NEW: Thalamic Reticular Nucleus (TRN) Searchlight ---
class TRNGate:
    def __init__(self):
        # Topographic Sectors: 0=Sensory(Visual/Whisker), 1=Memory(Hippocampal)
        self.sectors = np.array([-65.0, -65.0]) # Membrane potentials
        self.h_gates = np.array([0.0, 0.0])     # T-channel inactivation
        self.modes = ["TONIC", "TONIC"]
        
        # Lateral Inhibition Weight
        self.W_lat = 0.5 
    
    def step(self, arousal_level, amygdala_drive, pfc_attention, dt=0.05):
        # Inputs:
        # arousal_level: Global modulation (NE/ACh)
        # amygdala_drive: Bottom-up override (Fear) -> Targets Sector 0 (Sensory)
        # pfc_attention: Top-down goal -> Targets Sector 1 (Memory/Internal) usually, or 0
        
        # 1. Calculate Inputs for each sector
        # Sector 0 (Sensory) gets Amygdala drive strongly
        input_0 = (arousal_level * 5.0) + (amygdala_drive * 10.0) 
        # Sector 1 (Memory) gets PFC drive (we'll assume pfc_attention favors memory for now)
        input_1 = (arousal_level * 5.0) + (pfc_attention * 5.0)
        
        inputs = np.array([input_0, input_1])
        
        # 2. Update Dynamics (with Lateral Inhibition)
        new_potentials = np.copy(self.sectors)
        
        for i in range(2):
            # Self-excitation + Input - Inhibition from Neighbor
            neighbor = 1 - i
            inhibition = 0.0
            if self.sectors[neighbor] > -55.0: # If neighbor is firing
                inhibition = self.W_lat * (self.sectors[neighbor] + 55.0)
            
            target_v = -65.0 + inputs[i] - inhibition
            
            # Clamp target for sleep/wake bounds
            if arousal_level < 0.2: target_v = -75.0
            
            # Membrane Update
            new_potentials[i] += (target_v - self.sectors[i]) * 0.1
            
            # T-Channel Gate Dynamics (h)
            h_inf = 1.0 / (1.0 + np.exp((new_potentials[i] + 70) / 4.0))
            self.h_gates[i] += ((h_inf - self.h_gates[i]) / 20.0) * dt
            
            # Mode Switching
            if self.h_gates[i] > 0.6 and new_potentials[i] > -65.0:
                self.modes[i] = "BURST" # Inhibits relay
            elif new_potentials[i] > -65.0:
                self.modes[i] = "TONIC" # Relays info
                
        self.sectors = new_potentials
        return self.modes

# --- NEW: PVLV & PBWM ARCHITECTURE ---
class PVLV_Learning:
    def __init__(self):
        # Primary Value (PV): The Amygdala (learns immediate reward)
        self.w_pv = 0.0 
        # Learned Value (LV): The Ventral Striatum (learns predictive cues)
        self.w_lv = 0.0
        # Learning rates
        self.alpha_pv = 0.1
        self.alpha_lv = 0.05
        
    def step(self, sensory_drive, reward_present):
        # 1. PV System (Rescorla-Wagner on Primary Reward)
        pv_prediction = self.w_pv * sensory_drive
        pv_error = reward_present - pv_prediction
        self.w_pv += self.alpha_pv * pv_error * sensory_drive
        
        # 2. LV System (Predicts the PV)
        # In a full simulation, this needs a trace. Simplified here:
        # It tries to equal the PV expectation
        lv_prediction = self.w_lv * sensory_drive
        lv_error = pv_prediction - lv_prediction
        self.w_lv += self.alpha_lv * lv_error * sensory_drive
        
        # Total Dopamine = PV_Error (at reward time) + LV_Prediction (at cue time)
        # This allows dopamine to appear BEFORE the reward (anticipation)
        dopamine = pv_error + lv_prediction
        return dopamine

class PFC_Stripe:
    def __init__(self, id):
        self.id = id
        self.memory = np.array([0.0, 0.0]) # Stores a vector (e.g., Goal Location)
        self.is_locked = False # Maintenance vs Update
        
    def update(self, input_vector, gate_signal):
        # gate_signal: >0 means GO (Update), <0 means NOGO (Maintain)
        if gate_signal > 0:
            self.memory = input_vector # Update memory
            self.is_locked = False
        else:
            self.is_locked = True # Keep old memory
        return self.memory

# REPLACES OLD BASAL GANGLIA
class BasalGanglia:
    def __init__(self):
        # Weights for gating logic (State -> Gate)
        self.w_gate = np.random.rand(3) # Weights for [Sensory, Memory, Drive]
        
        # Action selection weights (State -> Motor)
        self.w_motor = np.ones(3) 
        
        self.pvlv = PVLV_Learning()
        
    def select_action_and_gate(self, sensory_vec, memory_vec, drive_level, reward, learning_rate_mod):
        # 1. Calculate Dopamine (PVLV)
        # Using magnitude of sensory input as "cue strength"
        sensory_mag = np.linalg.norm(sensory_vec)
        dopamine = self.pvlv.step(sensory_mag, reward)
        
        # 2. Gating Decision (Go vs NoGo)
        # Input state context
        context = np.array([np.linalg.norm(sensory_vec), np.linalg.norm(memory_vec), drive_level])
        
        # Dot product to decide gating
        gate_activation = np.dot(self.w_gate, context) + dopamine
        gate_signal = 1.0 if gate_activation > 0.5 else -1.0 # Threshold
        
        # 3. Motor Decision
        # Combines Sensory (Reactive) and Memory (Goal-Directed)
        # If Dopamine is high, weight Sensory more (Exploit). If low, weight Memory (Explore/Plan).
        
        w_sensory = self.w_motor[0] * (1.0 + dopamine)
        w_memory  = self.w_motor[1] * (1.0 - dopamine)
        
        motor_out = (sensory_vec * w_sensory) + (memory_vec * w_memory)
        
        # 4. Learning (Plasticity)
        # Update gating weights based on RPE (Dopamine)
        # If dopamine was positive, reinforce the choice we made
        if learning_rate_mod > 0.1:
            self.w_gate += 0.01 * dopamine * context * learning_rate_mod
            self.w_motor += 0.01 * dopamine * learning_rate_mod
            
            # Clip
            self.w_gate = np.clip(self.w_gate, -1.0, 1.0)
            self.w_motor = np.clip(self.w_motor, 0.0, 5.0)
            
        return motor_out, gate_signal, dopamine

# --- NEW: HIPPOCAMPAL TIME CELLS ---
class TimeCellPopulation:
    def __init__(self, n_cells=20):
        self.n_cells = n_cells
        self.cells = np.zeros(n_cells)
        # Each cell peaks at a different time (logarithmic scale typically, linear here for simplicity)
        self.peaks = np.linspace(0, 100, n_cells) # Time constants in frames
        self.sigmas = np.linspace(5, 20, n_cells) # Width of the time field
        self.timer = 0
        self.active = False
        
    def reset(self):
        self.timer = 0
        self.active = True
        self.cells.fill(0.0)
        
    def step(self):
        if not self.active: return np.zeros(self.n_cells)
        
        self.timer += 1
        # Gaussian receptive fields in Time
        # r_i(t) = exp( - (t - mu_i)^2 / 2sigma_i^2 )
        for i in range(self.n_cells):
            self.cells[i] = np.exp( -((self.timer - self.peaks[i])**2) / (2 * self.sigmas[i]**2) )
            
        return self.cells

class DendriticCluster:
    def __init__(self):
        # Replaced old Agents with new Architecture
        self.pfc = PFC_Stripe(0)
        self.basal_ganglia = BasalGanglia() # New Class
        
        self.soma = MicrotubuleSimulator2D("Executive Soma", length_points=32)
        self.astrocyte = Astrocyte()
        self.trn = TRNGate()
        self.time_cells = TimeCellPopulation()
        
        # STC Variables
        self.synaptic_tags = np.zeros(3) # [Sensory, Memory, Drive] tags
        self.prp_level = 0.0 # Plasticity Related Proteins
        
        # (Keep MT_Theta, Grid, HD setups) ...
        self.mt_theta = MicrotubuleSimulator2D("Theta Memory", length_points=32)
        self.mt_theta.H_BAR = 6.0 
        self.grid_k = [np.array([np.cos(0), np.sin(0)]), np.array([np.cos(np.pi/3), np.sin(np.pi/3)]), np.array([np.cos(2*np.pi/3), np.sin(2*np.pi/3)])]
        self.grid_phases = np.array([0.0, 0.0, 0.0]) 
        self.grid_scale = 0.8 
        self.hd_cell_count = 36 
        self.hd_cells = self._create_activity_bump(self.hd_cell_count, 0)
        self.hd_integration_weight = 2.0 
        self.place_memories = [] 
        self.weights = {"amygdala": 1.0, "striatum": 1.0, "hippocampus": 1.0}
        self.load_latest_memory()

    def _create_activity_bump(self, size, position):
        x = np.arange(size)
        bump = np.exp(-0.5 * np.minimum((x - position)**2, (size - np.abs(x - position))**2) / (size/10)**2)
        return bump / np.sum(bump)

    def update_hd_cells(self, angular_velocity):
        shift = -angular_velocity * self.hd_integration_weight * self.hd_cell_count / (2 * np.pi)
        f_hd = np.fft.fft(self.hd_cells)
        freqs = np.fft.fftfreq(self.hd_cell_count)
        f_hd *= np.exp(-1j * 2 * np.pi * freqs * shift)
        self.hd_cells = np.real(np.fft.ifft(f_hd))
        self.hd_cells += 0.01 * self._create_activity_bump(self.hd_cell_count, np.argmax(self.hd_cells))
        self.hd_cells /= np.sum(self.hd_cells)

    def get_head_direction(self): return np.argmax(self.hd_cells) * (2 * np.pi / self.hd_cell_count)

    def update_grid_cells(self, velocity, head_direction):
        rotation_matrix = np.array([[np.cos(head_direction), -np.sin(head_direction)], [np.sin(head_direction),  np.cos(head_direction)]])
        for i in range(3):
            rotated_k = rotation_matrix @ self.grid_k[i]
            proj = np.dot(velocity, rotated_k)
            drift = np.random.normal(0, 0.001)
            self.grid_phases[i] += (proj * self.grid_scale) + drift
            self.grid_phases[i] %= (2 * np.pi) 

    def get_grid_visualization(self):
        size = 32
        x = np.linspace(-np.pi, np.pi, size); y = np.linspace(-np.pi, np.pi, size)
        X, Y = np.meshgrid(x, y)
        W1 = np.cos(X * np.cos(0) + Y * np.sin(0) + self.grid_phases[0])
        W2 = np.cos(X * np.cos(np.pi/3) + Y * np.sin(np.pi/3) + self.grid_phases[1])
        W3 = np.cos(X * np.cos(2*np.pi/3) + Y * np.sin(2*np.pi/3) + self.grid_phases[2])
        return (W1 + W2 + W3 + 3) / 6.0 

    def add_place_memory(self):
        if len(self.place_memories) >= 200: self.place_memories.pop(0)
        self.place_memories.append(self.grid_phases.copy())
        
    def load_latest_memory(self):
        if os.path.exists("evolution_log.csv"):
            try:
                with open("evolution_log.csv", "r") as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        last_line = lines[-1].strip().split(',')
                        self.weights["amygdala"] = float(last_line[2])
                        self.weights["striatum"] = float(last_line[3])
                        self.weights["hippocampus"] = float(last_line[4])
            except: pass
    
    def save_memory(self):
         with open("brain_genome.json", "w") as f: json.dump(self.weights, f)

    def update_stc(self, activity_vector, reward_signal, dt=0.05):
        # 1. Tagging: Activity creates transient tags
        # Decay tags
        self.synaptic_tags *= 0.95 
        # Set new tags based on activity (Hebbian-like)
        self.synaptic_tags += 0.1 * np.abs(activity_vector)
        self.synaptic_tags = np.clip(self.synaptic_tags, 0, 1.0)
        
        # 2. Protein Synthesis (PRPs)
        # Triggered by strong events (Reward > 0.5 or strong Dopamine)
        if reward_signal > 0.5:
            self.prp_level = 1.0 # Flood the system
        else:
            self.prp_level *= 0.98 # Decay slowly
            
        # 3. Capture (Consolidation)
        # If Tag + PRP exists, permanent weight change happens
        # We simulate this by returning a "Consolidation Factor"
        consolidation = self.synaptic_tags * self.prp_level
        return consolidation

    def process_votes(self, frustration, dopamine, rat_vel, reward_signal, danger_level, pheromones_len, head_direction, vision_data, anesthetic_level=0.0):
        # 1. Update Spatial/Metabolic/TRN (Keep existing logic)
        self.update_grid_cells(rat_vel, head_direction)
        neural_activity = np.linalg.norm(rat_vel) + frustration
        atp_availability, ext_lactate = self.astrocyte.step(neural_activity)
        
        arousal = (dopamine * 0.5) + (frustration * 0.5)
        if atp_availability < 0.2: arousal = 0.0
        
        amygdala_drive = danger_level
        pfc_drive = frustration
        trn_modes = self.trn.step(arousal, amygdala_drive, pfc_drive)
        
        gain_sensory = 1.0 if trn_modes[0] == "TONIC" else 0.05
        gain_memory  = 1.0 if trn_modes[1] == "TONIC" else 0.05
        
        # --- NEW: Time Cell Update ---
        # Reset timer on significant events (e.g., Reward or Start of episode)
        if reward_signal > 0: self.time_cells.reset()
        temporal_context = self.time_cells.step()
        
        # 2. CONSTRUCT INPUT VECTORS
        # Sensory Vector: Derived from Vision (Targets) + Danger
        # Ideally, use the vision buffer. For simplicity, we create a vector towards the target if seen.
        sensory_vec = np.array([0.0, 0.0])
        closest_dist = 999.0
        for ray in vision_data:
            if ray['type'] == 3: # Target
                if ray['dist'] < closest_dist:
                    closest_dist = ray['dist']
                    sensory_vec = np.array([np.cos(ray['angle']), np.sin(ray['angle'])])
        
        # Apply Sensory Gain (TRN)
        sensory_vec *= gain_sensory
        
        # 3. PBWM / PVLV STEP (With Time)
        # The PFC holds a "Goal" vector.
        current_memory = self.pfc.memory * gain_memory
        
        # Plasticity Gate from Lactate
        plasticity_mod = ext_lactate / 0.5
        
        # Run Basal Ganglia
        final_vector, gate_signal, internal_dopamine = self.basal_ganglia.select_action_and_gate(
            sensory_vec, current_memory, frustration, reward_signal, plasticity_mod
        )
        
        # --- NEW: Synaptic Tagging & Capture ---
        # The 'context' used in BG was [Sensory, Memory, Drive]
        # We approximate the activity vector for STC
        activity_proxy = np.array([np.linalg.norm(sensory_vec), np.linalg.norm(current_memory), frustration])
        consolidation_vector = self.update_stc(activity_proxy, reward_signal)
        
        # Apply consolidation to BG weights (Long-term storage)
        # This makes the "temporary" learning from dopamine permanent if PRPs are present
        if np.max(consolidation_vector) > 0.1:
            self.basal_ganglia.w_gate += 0.001 * consolidation_vector * internal_dopamine
            
        # 4. Update PFC
        # If gate_signal is positive, the BG allows the PFC to update its goal to the current sensory input
        # (e.g., "I see cheese, make that my goal")
        self.pfc.update(sensory_vec, gate_signal)
        
        # 5. Soma & MT (Keep existing logic)
        angle = np.arctan2(final_vector[1], final_vector[0])
        if angle < 0: angle += 2*np.pi
        soma_input = np.zeros((self.soma.Ny, self.soma.Nx))
        target_idx = int((angle / (2*np.pi)) * self.soma.Nx) % 13
        soma_input[:, target_idx] += 5.0
        
        pump_map = soma_input * atp_availability
        d_soma, c_soma = self.soma.step(pump_map, lfp_signal=0.0, anesthetic_conc=anesthetic_level)
        
        theta_pump = (0.5 + (pheromones_len/200.0)) * atp_availability
        d_theta, c_theta = self.mt_theta.step(theta_pump, lfp_signal=0.0, anesthetic_conc=anesthetic_level)

        # Return updated dopamine from PVLV for the UI
        return d_soma, d_theta, self.get_grid_visualization(), final_vector, self.astrocyte.glycogen, self.astrocyte.neuronal_atp

class Predator:
    def __init__(self, pos): self.pos = np.array(pos); self.vel = np.array([0.0, 0.0]); self.speed = 0.35 
    def hunt(self, rat_pos, grid, map_size):
        diff = rat_pos - self.pos; dist = np.linalg.norm(diff)
        if dist > 0:
            desired_vel = (diff / dist) * self.speed
            next_pos = self.pos + desired_vel
            nx, ny = int(next_pos[0]), int(next_pos[1])
            if 0 <= nx < map_size and 0 <= ny < map_size:
                if grid[ny, nx] == 0: self.pos = next_pos
                else:
                    if grid[ny, int(self.pos[0])] == 0: self.pos[1] = next_pos[1]
                    elif grid[int(self.pos[1]), nx] == 0: self.pos[0] = next_pos[0]

class Cerebellum:
    def __init__(self):
        self.predicted_pos = np.array([0.0, 0.0])
        self.correction_vector = np.array([0.0, 0.0])
        self.learning_rate = 0.1 # How quickly the cerebellum adapts

    def predict(self, current_pos, efference_copy):
        # The cerebellum's forward model predicts where the rat *should* go
        # based on the motor command (efference copy)
        self.predicted_pos = current_pos + (efference_copy * 0.15) # Assuming efference_copy scales similarly to rat_vel update

    def update(self, actual_pos):
        # Compare prediction with actual sensory input (actual_pos)
        error = actual_pos - self.predicted_pos
        # Update the correction vector
        self.correction_vector = self.correction_vector * (1 - self.learning_rate) + error * self.learning_rate

# --- SIMULATION STATE ---
class SimulationState:
    def __init__(self):
        self.brain = DendriticCluster()
        self.map_size = 40; self.rat_pos = np.array([3.0, 3.0]); self.rat_vel = np.array([0.0, 0.6])
        self.targets = [] # Changed from single target to a list of targets
        self.predator = Predator([35.0, 3.0])
        self.cerebellum = Cerebellum() # Instantiate the Cerebellum
        self.score = 0; self.deaths = 0; self.frustration = 0.0; self.dopamine = 0.2
        self.serotonin = 0.5; self.metabolic_rate = 0.0003 
        self.state = "AWAKE"; self.dream_scans = []; self.phantom_trace = []; self.pheromones = [] 
        self.dream_timer = 0; self.panic = 0; self.config = {"speed": 1, "sensitivity": 0.05, "anesthetic": 0.0}
        
        self.generation = 1; self.best_fitness = 0.0; self.current_fitness = 0.0; self.frames_alive = 0
        self.best_genome = copy.deepcopy(self.brain.weights)
        self.run_history = []  
        self.genes_being_tested = copy.deepcopy(self.brain.weights)
        self.stable_genome = copy.deepcopy(self.brain.weights)
        
        # Add Vibrissal Variables
        self.whisk_angle = 0.0
        self.whisk_phase = 0.0
        self.whisk_freq = 0.8  # ~8Hz relative to dt
        self.whisk_amp = np.pi / 4.0 # 45 degree sweep
        self.whisker_hits = [False, False] # Left bank, Right bank contact
        self.vision_buffer = []

        # Add Rat Heading
        self.rat_heading = np.pi / 2 # Initial heading to match initial velocity [0, 0.6]
        
        if not os.path.exists("evolution_log.csv"):
            with open("evolution_log.csv", "w") as f: f.write("Generation,Fitness,Amygdala,Striatum,Hippocampus\n")
        else:
            with open("evolution_log.csv", "r") as f:
                lines = f.readlines()
                if len(lines) > 1: self.generation = int(lines[-1].split(',')[0]) + 1

        self.occupancy_grid = np.zeros((self.map_size, self.map_size), dtype=int)
        self.generate_map()

    def generate_map(self):
        while True:
            self.occupancy_grid = np.random.choice([0, 1], size=(self.map_size, self.map_size), p=[0.6, 0.4])
            for _ in range(5):
                new_grid = self.occupancy_grid.copy()
                for y in range(1, self.map_size-1):
                    for x in range(1, self.map_size-1):
                        neighbors = np.sum(self.occupancy_grid[y-1:y+2, x-1:x+2])
                        if neighbors > 4: new_grid[y,x] = 1
                        elif neighbors < 4: new_grid[y,x] = 0
                self.occupancy_grid = new_grid
            self.occupancy_grid[0,:] = 1; self.occupancy_grid[-1,:] = 1
            self.occupancy_grid[:,0] = 1; self.occupancy_grid[:,-1] = 1
            self.occupancy_grid[1:6, 1:6] = 0; self.occupancy_grid[34:39, 34:39] = 0
            self.occupancy_grid[1:6, 34:39] = 0
            
            test_grid = self.occupancy_grid.copy(); start = (3, 3)
            queue = [start]; test_grid[start] = 2; reachable_tiles = [start]
            while queue:
                cx, cy = queue.pop(0)
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nx, ny = cx+dx, cy+dy
                    if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                        if test_grid[ny, nx] == 0:
                            test_grid[ny, nx] = 2; queue.append((nx, ny)); reachable_tiles.append((nx, ny))
            
            if len(reachable_tiles) > 200:
                self.rat_pos = np.array([3.0, 3.0])
                reachable_tiles.sort(key=lambda p: (p[0]-3)**2 + (p[1]-3)**2)
                
                self.targets = [] # Clear existing targets
                for _ in range(3): # Spawn 3 targets
                    self.targets.append(self._spawn_single_target())

                far_tiles = [t for t in reachable_tiles if (t[0]-3)**2 + (t[1]-3)**2 > 200]
                if len(far_tiles) > 0:
                    ridx = np.random.randint(len(far_tiles))
                    p_spawn = far_tiles[ridx]
                    self.predator = Predator([float(p_spawn[0]), float(p_spawn[1])])
                    self.pheromones = []
                    self.brain.astrocyte = Astrocyte() # Also reset on new map
                    break 

    def check_collision(self, pos):
        # Helper to check a single grid cell
        def is_wall(x, y):
            ix, iy = int(x), int(y)
            if ix < 0 or ix >= self.map_size or iy < 0 or iy >= self.map_size: return True
            return self.occupancy_grid[iy, ix] == 1

        # 1. Check Center
        if is_wall(pos[0], pos[1]): return True
        
        # 2. Check "Body Radius" (Prevent visual clipping)
        # We check 4 points around the center at a radius of 0.3
        radius = 0.3
        if is_wall(pos[0] + radius, pos[1]): return True
        if is_wall(pos[0] - radius, pos[1]): return True
        if is_wall(pos[0], pos[1] + radius): return True
        if is_wall(pos[0], pos[1] - radius): return True
        
        return False
    
    def _spawn_single_target(self):
        while True:
            tx, ty = np.random.randint(1, self.map_size - 1), np.random.randint(1, self.map_size - 1)
            if self.occupancy_grid[ty, tx] == 0:
                # Ensure the target is reachable and somewhat far from the rat's spawn
                test_grid = self.occupancy_grid.copy()
                start = (int(self.rat_pos[0]), int(self.rat_pos[1]))
                queue = [start]
                test_grid[start[1], start[0]] = 2  # Mark as visited
                
                # BFS to check reachability
                reachable = False
                steps = 0
                max_steps = self.map_size * self.map_size # Prevent infinite loops
                
                while queue and steps < max_steps:
                    cx, cy = queue.pop(0)
                    if cx == tx and cy == ty:
                        reachable = True
                        break
                    steps += 1
                    for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                        nx, ny = cx+dx, cy+dy
                        if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                            if test_grid[ny, nx] == 0:
                                test_grid[ny, nx] = 2
                                queue.append((nx, ny))
                
                if reachable:
                    return np.array([float(tx), float(ty)])

    def cast_ray(self, angle):
        dx, dy = np.cos(angle), np.sin(angle)
        
        # --- NEW LOGIC START ---
        min_dist = 15.0
        obj_type = 0
        
        # 1. Check Walls first to establish occlusion depth
        for d in np.linspace(0, 15.0, 30): 
            cx = self.rat_pos[0] + dx * d
            cy = self.rat_pos[1] + dy * d
            if self.check_collision([cx, cy]): 
                min_dist = d
                obj_type = 1
                break # Stop at first wall hit
        
        # 2. Check Predator (Only visible if closer than wall)
        to_predator = self.predator.pos - self.rat_pos
        dist_predator = np.linalg.norm(to_predator)
        if dist_predator > 0:
            pred_angle = np.arctan2(to_predator[1], to_predator[0])
            # Handle angle wrapping if necessary, or keep simple abs check
            if abs(angle - pred_angle) < 0.1: 
                if dist_predator < min_dist: # Only if closer than current min_dist
                    min_dist = dist_predator
                    obj_type = 2
        
        # 3. Check Targets (Only visible if closer than wall/predator)
        for target in self.targets:
            to_target = target - self.rat_pos
            dist_target = np.linalg.norm(to_target)
            if dist_target > 0:
                target_angle = np.arctan2(to_target[1], to_target[0])
                if abs(angle - target_angle) < 0.1: # Within a small angular tolerance
                    if dist_target < min_dist: # Only if closer than current min_dist
                        min_dist = dist_target
                        obj_type = 3
                
        return min_dist, obj_type
        # --- NEW LOGIC END ---

    def process_vision(self):
        self.vision_buffer = []
        heading = np.arctan2(self.rat_vel[1], self.rat_vel[0])
        fov = np.pi * (2/3) # 120 degrees
        
        for i in range(12):
            angle_offset = (i / 11.0 - 0.5) * fov
            ray_angle = heading + angle_offset
            dist, obj_type = self.cast_ray(ray_angle)
            self.vision_buffer.append({'dist': dist, 'type': obj_type, 'angle': ray_angle})

    def update_whiskers(self):
        # 1. Oscillate Phase
        self.whisk_phase += self.whisk_freq
        # Simple Harmonic Motion: Cosine wave for sweep
        self.whisk_angle = np.sin(self.whisk_phase) * self.whisk_amp
        
        # 2. Define Whisker Banks (Relative angles to head)
        # Left bank (-45 deg base), Right bank (+45 deg base) + oscillation
        angles = [
            (-np.pi/4) + self.whisk_angle, # Left
            (np.pi/4) - self.whisk_angle   # Right (Anti-phase or Synchronous? We use Synchronous for simplicity)
        ]
        
        heading = np.arctan2(self.rat_vel[1], self.rat_vel[0])
        self.whisker_hits = [False, False]
        whisker_length = 5.0 # Shorter than vision, requires close proximity
        
        # 3. Check Collisions (Physical Touch)
        for i, angle in enumerate(angles):
            # Calculate Tip Position
            global_angle = heading + angle
            tip_x = self.rat_pos[0] + np.cos(global_angle) * whisker_length
            tip_y = self.rat_pos[1] + np.sin(global_angle) * whisker_length
            
            # Ray trace the shaft of the whisker (check midpoint and tip)
            mid_x = self.rat_pos[0] + np.cos(global_angle) * (whisker_length * 0.5)
            mid_y = self.rat_pos[1] + np.sin(global_angle) * (whisker_length * 0.5)
            
            if self.check_collision([mid_x, mid_y]) or self.check_collision([tip_x, tip_y]):
                self.whisker_hits[i] = True

    def dream(self):
        self.dream_timer += 1
        if self.dream_timer > 20:
            rand_angle = np.random.rand() * 2 * np.pi
            self.rat_vel = np.array([np.cos(rand_angle), np.sin(rand_angle)]) * 0.5
            self.state = "AWAKE"; self.dream_timer = 0; self.frustration = 0.0
            return True
        return False
        
    def die(self):
        fitness = (self.score * 100) + (self.frames_alive * 0.01)
        self.run_history.append(fitness)
        BATCH_SIZE = 5 
        
        with open("evolution_log.csv", "a") as f:
            f.write(f"{self.generation},{fitness:.2f},{self.brain.weights['amygdala']:.3f},{self.brain.weights['striatum']:.3f},{self.brain.weights['hippocampus']:.3f}\n")
        
        if len(self.run_history) >= BATCH_SIZE:
            avg_fitness = sum(self.run_history) / len(self.run_history)
            print(f"Gen {self.generation} Batch Avg: {avg_fitness:.2f} vs Best: {self.best_fitness:.2f}")
            if avg_fitness > self.best_fitness:
                self.best_fitness = avg_fitness
                self.stable_genome = copy.deepcopy(self.genes_being_tested) 
                self.brain.save_memory()
                print(">>> NEW BEST GENOME SAVED <<<")
            
            self.brain.weights = copy.deepcopy(self.stable_genome)
            mutation_rate = 0.2
            for k in self.brain.weights:
                change = 1.0 + np.random.normal(0, mutation_rate) 
                self.brain.weights[k] = max(0.1, min(10.0, self.brain.weights[k] * change))
                
            self.genes_being_tested = copy.deepcopy(self.brain.weights)
            self.run_history = [] 
            self.generation += 1
        
        self.deaths += 1; self.frames_alive = 0
        self.rat_pos = np.array([3.0, 3.0]); self.rat_vel = np.array([0.0, 0.6])
        self.pheromones = []; self.frustration = 0; self.panic = 0
        
        # --- CRITICAL FIXES ---
        self.brain.astrocyte = Astrocyte() # Reset metabolism
        self.brain.trn = TRNGate()         # Reset Sleep/Wake cycle
        self.brain.basal_ganglia = BasalGanglia() # Optional: Reset learned weights? 
        # usually evolution handles weights, but we should reset short-term accumulators if any. 
        # Since your BasalGanglia only holds weights, we might actually WANT to keep them 
        # if they are part of the genome. If they are learned 'per life', reset them.
        # For now, let's keep weights (Evolutionary) but ensure TRN is awake.
        
        # CLEAR SHORT TERM MEMORIES ON DEATH (New Generation)
        self.brain.place_memories = []

        while True:
            tx, ty = np.random.randint(1, 39), np.random.randint(1, 39)
            if self.occupancy_grid[ty, tx] == 0:
                dist = (tx-3)**2 + (ty-3)**2
                if dist > 200: self.predator = Predator([float(tx), float(ty)]); break

sim = SimulationState()

@app.route('/')
def index(): return render_template('index.html')

@app.route('/step', methods=['POST'])
def step():
    req_data = request.json or {}
    iterations = req_data.get("batch_size", int(sim.config["speed"]))
    
    # NEW: Get anesthetic level from config
    anesthetic_level = float(sim.config.get("anesthetic", 0.0))
    
    for i in range(iterations):
        sim.update_whiskers()
        sim.process_vision()
        sim.frames_alive += 1

        # REPLACEMENT for sim.energy logic
        # We use the Brain's Astrocytic state
        current_atp = sim.brain.astrocyte.neuronal_atp

        # Logic: If ATP is 0, the rat dies of excitotoxicity/failure
        dist_cat = np.linalg.norm(sim.rat_pos - sim.predator.pos)
        if current_atp <= 0.01 or dist_cat < 1.0:
            sim.die()
            return jsonify({ "rat": sim.rat_pos.tolist(), "targets": [t.tolist() for t in sim.targets], "predator": sim.predator.pos.tolist(), "brain": {}, "stats": {"frustration": 0, "score": sim.score, "status": "DEAD", "energy": 0, "atp": 0, "deaths": sim.deaths, "generation": sim.generation}, "reset_visuals": True, "whiskers": {"angle": sim.whisk_angle, "hits": sim.whisker_hits}, "vision": sim.vision_buffer})

        sim.dopamine = max(0.1, sim.dopamine * 0.995)
        if sim.frustration < 0.1: sim.serotonin = min(1.0, sim.serotonin + 0.005)
        
        # Check if WHOLE brain is in burst (Sleep) or just specific sectors
        # For simulation status, we say "MICROSLEEP" only if Sensory is blocked
        trn_modes = sim.brain.trn.modes
        
        if trn_modes[0] == "BURST" and trn_modes[1] == "BURST":
             if i == iterations - 1:
                soma_density, d_theta, d_grid, final_decision, glycogen_level, atp_level = sim.brain.process_votes(
                    sim.frustration, sim.dopamine, sim.rat_vel, 0, 0, len(sim.pheromones), 
                    sim.rat_heading, sim.vision_buffer, anesthetic_level=anesthetic_level
                )
                return jsonify({ "rat": sim.rat_pos.tolist(), "targets": [t.tolist() for t in sim.targets], "predator": sim.predator.pos.tolist(), 
                    "brain": {"soma": soma_density.tolist(), "theta": d_theta.tolist(), "grid": d_grid.tolist(), "hd": sim.brain.hd_cells.tolist()},
                    "cerebellum": {"predicted_pos": sim.cerebellum.predicted_pos.tolist(), "correction_vector": sim.cerebellum.correction_vector.tolist()},
                    "stats": {"frustration": sim.frustration, "score": sim.score, "status": "MICROSLEEP", "dopamine": sim.dopamine, "serotonin": sim.serotonin, "energy": glycogen_level, "atp": atp_level, "deaths": sim.deaths, "generation": sim.generation}, "phantom": sim.phantom_trace, "pheromones": sim.pheromones,
                    "whiskers": {"angle": sim.whisk_angle, "hits": sim.whisker_hits}, "vision": sim.vision_buffer})
             continue

        danger_level = 0.0
        if dist_cat < 10.0: danger_level = 1.0 - (dist_cat / 10.0)
        
        reward_signal = 0.0
        for idx, target_pos in enumerate(sim.targets):
             if np.linalg.norm(sim.rat_pos - target_pos) < 1.0:
                sim.score += 1; sim.frustration = 0; sim.dopamine = 1.0; 
                sim.brain.astrocyte.glycogen = 1.0 
                reward_signal = 1.0 
                sim.targets[idx] = sim._spawn_single_target()
                break 

        # --- UPDATE BRAIN ---
        new_heading = np.arctan2(sim.rat_vel[1], sim.rat_vel[0])
        angular_velocity = (new_heading - sim.rat_heading)
        if angular_velocity > np.pi: angular_velocity -= 2 * np.pi
        if angular_velocity < -np.pi: angular_velocity += 2 * np.pi
        sim.rat_heading = new_heading
        sim.brain.update_hd_cells(angular_velocity) # Assume angular_velocity is calculated as before
        head_direction = sim.brain.get_head_direction()

        # PASS ANESTHETIC HERE TOO
        soma_density, d_theta, d_grid, final_decision, glycogen_level, atp_level = sim.brain.process_votes(
            sim.frustration, sim.dopamine, sim.rat_vel, reward_signal, danger_level, 
            len(sim.pheromones), head_direction, sim.vision_buffer, anesthetic_level=anesthetic_level
        )
        
        sim.cerebellum.predict(sim.rat_pos, final_decision)
        final_decision_corrected = final_decision + sim.cerebellum.correction_vector
        
        sim.rat_vel = (sim.rat_vel * 0.85) + (final_decision_corrected * 0.15)
        sim.rat_vel = np.nan_to_num(sim.rat_vel) 

        tremor = np.random.randn(2) * 0.02; sim.rat_vel += tremor
        s = np.linalg.norm(sim.rat_vel)
        max_speed = 0.3 + (sim.dopamine * 0.2)
        if s > max_speed: sim.rat_vel = (sim.rat_vel / s) * max_speed
        
        new_pos = sim.rat_pos + sim.rat_vel
        
        hit = sim.check_collision(new_pos) 
        if hit:
            sim.rat_vel *= -0.5
            sim.frustration = min(1.0, sim.frustration + 0.1)
            sim.serotonin = max(0.1, sim.serotonin - 0.1)
        else:
            sim.rat_pos = new_pos
            sim.frustration = max(0.0, sim.frustration - 0.005)
        
        sim.cerebellum.update(sim.rat_pos)
            
        if np.random.rand() < 0.2:
            sim.pheromones.append(sim.rat_pos.tolist())
            if len(sim.pheromones) > 1000: sim.pheromones.pop(0)
    
    # Return state for last frame of batch
    return jsonify({"rat": sim.rat_pos.tolist(), "targets": [t.tolist() for t in sim.targets], "predator": sim.predator.pos.tolist(), 
                    "brain": {"soma": soma_density.tolist(), "theta": d_theta.tolist(), "grid": d_grid.tolist(), "hd": sim.brain.hd_cells.tolist()},
                    "cerebellum": {"predicted_pos": sim.cerebellum.predicted_pos.tolist(), "correction_vector": sim.cerebellum.correction_vector.tolist()},
                    "stats": {"frustration": sim.frustration, "score": sim.score, "status": "AWAKE" if "TONIC" in sim.brain.trn.modes else "MICROSLEEP", "dopamine": sim.dopamine, "serotonin": sim.serotonin, "energy": glycogen_level, "atp": atp_level, "deaths": sim.deaths, "generation": sim.generation}, "phantom": sim.phantom_trace, "pheromones": sim.pheromones,
                    "whiskers": {"angle": sim.whisk_angle, "hits": sim.whisker_hits}, "vision": sim.vision_buffer})

# (Routes for history, load_generation, reset, config remain identical)
@app.route('/history', methods=['GET'])
def history():
    data = []
    if os.path.exists("evolution_log.csv"):
        with open("evolution_log.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader: data.append(row)
    return jsonify(data)

@app.route('/load_generation', methods=['POST'])
def load_generation():
    gen_id = str(request.json.get("generation"))
    if os.path.exists("evolution_log.csv"):
        with open("evolution_log.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["Generation"] == gen_id:
                    sim.brain.weights["amygdala"] = float(row["Amygdala"])
                    sim.brain.weights["striatum"] = float(row["Striatum"])
                    sim.brain.weights["hippocampus"] = float(row["Hippocampus"])
                    sim.generate_map(); sim.deaths = 0
                    return jsonify({"status": "loaded", "gen": gen_id})
    return jsonify({"status": "error", "msg": "Gen not found"})

@app.route('/reset', methods=['POST'])
def reset():
    sim.generate_map()
    sim.frustration = 0; sim.dopamine = 0.2; sim.serotonin = 0.5; sim.deaths = 0; sim.generation = 1
    with open("evolution_log.csv", "w") as f: f.write("Generation,Fitness,Amygdala,Striatum,Hippocampus\n")
    walls = []
    for y in range(sim.map_size):
        for x in range(sim.map_size):
            if sim.occupancy_grid[y,x] == 1: walls.append([x, y, 1, 1])
    return jsonify({"status": "reset", "walls": walls})

@app.route('/config', methods=['POST'])
def config(): sim.config.update(request.json); return jsonify({"status": "updated"})

if __name__ == '__main__': app.run(debug=True, port=5000)
