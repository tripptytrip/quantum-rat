from __future__ import annotations

from flask import Flask, render_template, jsonify, request
import numpy as np
import json
import os
import copy
import csv
from threading import Lock
from typing import Any, Dict, List, Tuple

app = Flask(__name__)

# Serialize access to global simulation state (single-user demo safety)
sim_lock = Lock()

# --- PHYSICS ENGINE ---
COMPLEX_TYPE = np.complex128


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def angdiff(a: float, b: float) -> float:
    """Smallest absolute angular difference between angles a and b (radians)."""
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return abs(d)


def maybe_downsample(arr: np.ndarray, ds: int) -> np.ndarray:
    """Downsample a 2D array by stride ds (ds=1 => no-op)."""
    if ds <= 1:
        return arr
    if arr.ndim == 1:
        return arr[::ds]
    if arr.ndim == 2:
        return arr[::ds, ::ds]
    return arr


class MicrotubuleSimulator2D:
    def __init__(self, label, length_points=32, seam_shift=3, dt=0.01):
        self.label = label
        # In this model: X=13 (Circumference/Protofilaments), Y=Length
        self.Nx = 13
        self.Ny = length_points
        self.dt = dt
        self.seam_shift = seam_shift  # The "3-start helix" shift

        # Physics Constants (normalized)
        self.H_BAR = 1.0
        self.base_gamma = 1.5

        # State Arrays
        self.psi = np.zeros((self.Ny, self.Nx), dtype=COMPLEX_TYPE)
        self.superradiance = np.zeros((self.Ny, self.Nx), dtype=float)
        self.accumulated_Eg = 0.0

        # Phases: "ISOLATION" (Classical), "SUPERPOSITION" (Quantum), "COLLAPSE"
        self.phase = "ISOLATION"

        self.reset_wavefunction()

    def reset_wavefunction(self):
        noise_r = 0.1 * (np.random.rand(self.Ny, self.Nx) - 0.5)
        noise_i = 0.1 * (np.random.rand(self.Ny, self.Nx) - 0.5)
        self.psi = (noise_r + 1j * noise_i).astype(COMPLEX_TYPE)
        self.superradiance.fill(0.0)
        self.accumulated_Eg = 0.0
        self.phase = "ISOLATION"

    def get_helical_neighbor(self, grid: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """
        Neighbor grid shifted by dx (circumference) and dy (length).

        - X wraps (cylindrical) with seam shift correction.
        - Y is OPEN ENDED (no wrapping): values shifted past ends are zeroed.
        """
        # Shift Y
        shifted = np.roll(grid, dy, axis=0)

        # Open ends: zero the wrapped-in row
        if dy == 1:
            shifted[0, :] = 0.0
        elif dy == -1:
            shifted[-1, :] = 0.0

        # Shift X with helical seam correction
        if dx == 0:
            return shifted

        temp_grid = np.roll(shifted, dx, axis=1)

        # Correct the seam wrap column by shifting along Y
        if dx == 1:
            temp_grid[:, 0] = np.roll(temp_grid[:, 0], self.seam_shift)
            # after shifting seam column in Y, enforce open ends again
            temp_grid[0, 0] = 0.0
            temp_grid[-1, 0] = 0.0
        elif dx == -1:
            temp_grid[:, -1] = np.roll(temp_grid[:, -1], -self.seam_shift)
            temp_grid[0, -1] = 0.0
            temp_grid[-1, -1] = 0.0

        return temp_grid

    def step(self, pump_map: np.ndarray, lfp_signal: float = 0.0, anesthetic_conc: float = 0.0):
        anesthetic_conc = clamp(float(anesthetic_conc), 0.0, 1.0)

        # Anesthesia dampens input sensitivity
        effective_pump = pump_map * (1.0 - anesthetic_conc)

        # 1) Update Superradiance (pumping vs decay)
        decay_rate = 0.1
        self.superradiance = self.superradiance * (1 - decay_rate) + (effective_pump * 0.2)
        self.superradiance = np.clip(self.superradiance, 0.0, 1.0)

        # 2) Dipole coupling field
        neighbors = [
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 0.58),
            (0, -1, 0.58),
            (1, 1, 0.4),
            (-1, -1, 0.4),
        ]

        dipole_potential = np.zeros_like(self.psi, dtype=COMPLEX_TYPE)

        for dy, dx, coupling in neighbors:
            psi_neighbor = self.get_helical_neighbor(self.psi, dx, dy)
            effective_coupling = coupling
            if dx != 0:
                effective_coupling *= (1.0 - anesthetic_conc)
            dipole_potential += effective_coupling * psi_neighbor

        # 3) Phase Dynamics
        collapse_event = False
        avg_coherence = float(np.mean(self.superradiance))

        if avg_coherence < 0.2:
            # Classical-ish relaxation
            self.phase = "ISOLATION"
            d_psi = (np.real(dipole_potential) - 4 * self.psi) * 0.1
            self.psi += d_psi
            self.psi /= (np.max(np.abs(self.psi)) + 1e-9)
        else:
            if self.phase == "ISOLATION":
                self.phase = "SUPERPOSITION"
                self.accumulated_Eg = 0.0

            density = np.abs(self.psi) ** 2
            nonlinear = self.base_gamma * density * self.psi

            # Pump drives Schrödinger-like evolution
            d_psi = -1j * (dipole_potential + nonlinear) + (effective_pump * self.psi)
            self.psi += d_psi * self.dt

            # Update Eg based on *current* density (recompute)
            density_now = np.abs(self.psi) ** 2
            current_Eg = float(np.var(density_now) * 10.0)
            self.accumulated_Eg += current_Eg * self.dt

            if self.accumulated_Eg > self.H_BAR:
                self.phase = "COLLAPSE"
                collapse_event = True
                # Collapse decision uses the latest density
                mean_d = float(np.mean(density_now))
                if np.random.rand() > 0.5:
                    self.psi = np.where(density_now > mean_d, 1.0 + 0j, 0.0 + 0j)
                else:
                    self.reset_wavefunction()
                self.accumulated_Eg = 0.0
                self.phase = "ISOLATION"

        # Normalize
        norm = float(np.linalg.norm(self.psi))
        if norm > 0:
            self.psi /= norm

        # IMPORTANT FIX: viz density must be computed from the updated psi
        density_viz = np.abs(self.psi) ** 2
        viz_density = density_viz * 200.0

        return viz_density, collapse_event


class Astrocyte:
    def __init__(self):
        self.glycogen = 1.0
        self.astro_lactate = 0.5
        self.ext_lactate = 0.5
        self.neuron_lactate = 0.5
        self.neuronal_atp = 1.0
        self.glutamate_load = 0.0

        self.Vmax_MCT4 = 0.8
        self.Km_MCT4 = 4.0

        self.Vmax_MCT2 = 0.6
        self.Km_MCT2 = 0.5

    def step(self, firing_rate: float, dt: float = 0.05):
        firing_rate = float(max(0.0, firing_rate))

        # Demand signal
        demand = firing_rate * 0.1
        self.glutamate_load += demand

        # Glycogenolysis (from glycogen -> lactate in astrocyte)
        glycolysis_rate = 0.05 * self.glutamate_load * self.glycogen
        self.glycogen -= glycolysis_rate * dt
        self.astro_lactate += glycolysis_rate * dt

        # Transport astro -> ECS (MCT4)
        flux_astro_to_ecs = self.Vmax_MCT4 * (self.astro_lactate / (self.Km_MCT4 + self.astro_lactate + 1e-9))

        # Transport ECS -> neuron (MCT2)
        flux_ecs_to_neuron = self.Vmax_MCT2 * (self.ext_lactate / (self.Km_MCT2 + self.ext_lactate + 1e-9))

        self.astro_lactate -= flux_astro_to_ecs * dt
        self.ext_lactate += (flux_astro_to_ecs - flux_ecs_to_neuron) * dt
        self.neuron_lactate += flux_ecs_to_neuron * dt

        # OxPhos: neuron lactate -> ATP
        conversion_rate = 2.0 * self.neuron_lactate
        atp_production = conversion_rate * 3.0
        self.neuron_lactate -= conversion_rate * dt

        basal_metabolism = 0.005
        activity_cost = firing_rate * 0.05
        self.neuronal_atp += (atp_production - (basal_metabolism + activity_cost)) * dt

        # Decay demand
        self.glutamate_load *= 0.9

        # Clamp compartments (IMPORTANT stability)
        self.glycogen = float(np.clip(self.glycogen, 0.0, 1.0))
        self.neuronal_atp = float(np.clip(self.neuronal_atp, 0.0, 1.5))

        self.astro_lactate = float(max(0.0, self.astro_lactate))
        self.ext_lactate = float(max(0.0, self.ext_lactate))
        self.neuron_lactate = float(max(0.0, self.neuron_lactate))

        return self.neuronal_atp, self.ext_lactate


class TRNGate:
    def __init__(self):
        self.sectors = np.array([-65.0, -65.0])
        self.h_gates = np.array([0.0, 0.0])
        self.modes = ["TONIC", "TONIC"]
        self.W_lat = 0.5

    def step(self, arousal_level: float, amygdala_drive: float, pfc_attention: float, dt: float = 0.05):
        arousal_level = clamp(float(arousal_level), 0.0, 2.0)
        amygdala_drive = clamp(float(amygdala_drive), 0.0, 10.0)
        pfc_attention = clamp(float(pfc_attention), 0.0, 10.0)

        input_0 = (arousal_level * 5.0) + (amygdala_drive * 10.0)
        input_1 = (arousal_level * 5.0) + (pfc_attention * 5.0)

        inputs = np.array([input_0, input_1])
        new_potentials = np.copy(self.sectors)

        for i in range(2):
            neighbor = 1 - i
            inhibition = 0.0
            if self.sectors[neighbor] > -55.0:
                inhibition = self.W_lat * (self.sectors[neighbor] + 55.0)

            target_v = -65.0 + inputs[i] - inhibition
            if arousal_level < 0.2:
                target_v = -75.0

            new_potentials[i] += (target_v - self.sectors[i]) * 0.1

            h_inf = 1.0 / (1.0 + np.exp((new_potentials[i] + 70.0) / 4.0))
            self.h_gates[i] += ((h_inf - self.h_gates[i]) / 20.0) * dt

            if self.h_gates[i] > 0.6 and new_potentials[i] > -65.0:
                self.modes[i] = "BURST"
            elif new_potentials[i] > -65.0:
                self.modes[i] = "TONIC"

        self.sectors = new_potentials
        return self.modes


class PVLV_Learning:
    def __init__(self):
        self.w_pv = 0.0
        self.w_lv = 0.0
        self.alpha_pv = 0.1
        self.alpha_lv = 0.05

    def step(self, sensory_drive: float, reward_present: float):
        sensory_drive = float(max(0.0, sensory_drive))
        reward_present = float(max(0.0, reward_present))

        pv_prediction = self.w_pv * sensory_drive
        pv_error = reward_present - pv_prediction
        self.w_pv += self.alpha_pv * pv_error * sensory_drive

        lv_prediction = self.w_lv * sensory_drive
        lv_error = pv_prediction - lv_prediction
        self.w_lv += self.alpha_lv * lv_error * sensory_drive

        dopamine = pv_error + lv_prediction
        return float(dopamine)


class VisionCortex:
    def __init__(self):
        self.ema_salience = 0.0

    def encode(self, vision_rays: List[Dict[str, Any]]) -> Dict[str, Any]:
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
            
            if typ in salience_weights:
                salience = salience_weights[typ] * (1.0 / (dist + 1e-9))
                if salience > max_salience:
                    max_salience = salience

                if typ == 3 and salience > max_target_salience:
                    max_target_salience = salience
                    salient_target_ray = ray

            if typ == 1: wall_dist = min(wall_dist, dist)
            if typ == 2: predator_dist = min(predator_dist, dist)
            if typ == 3: target_dist = min(target_dist, dist)

        vis_vec = np.array([0.0, 0.0])
        if salient_target_ray:
            angle = salient_target_ray.get("angle", 0.0)
            vis_vec = np.array([np.cos(angle), np.sin(angle)])

        novelty = abs(max_salience - self.ema_salience)
        self.ema_salience = 0.9 * self.ema_salience + 0.1 * max_salience

        features = np.array([
            clamp(1.0 - wall_dist / 15.0, 0.0, 1.0),
            clamp(1.0 - predator_dist / 15.0, 0.0, 1.0),
            clamp(1.0 - target_dist / 15.0, 0.0, 1.0),
            clamp(novelty, 0.0, 1.0)
        ])
        
        return {
            "vis_vec": vis_vec,
            "features": features
        }

class SomatoCortex:
    def __init__(self):
        self.ema_pain = 0.0

    def encode(self, whisker_hits, collision: bool) -> Dict[str, Any]:
        touch = 1.0 if whisker_hits and any(whisker_hits) else 0.0
        pain = 1.0 if collision else 0.0
        
        self.ema_pain = max(pain, self.ema_pain * 0.9)
        
        return {
            "touch": touch,
            "pain": self.ema_pain
        }

class Thalamus:
    def __init__(self):
        self.ema_gain = 1.0

    def relay(self,
              vis: Dict[str, Any],
              som: Dict[str, Any],
              topdown_attention: float = 0.0,
              relay_gain: float = 1.0
              ) -> Dict[str, Any]:
        
        sensory_gain = relay_gain * clamp(0.8 + 0.04 * topdown_attention, 0.5, 1.5)
        
        self.ema_gain = 0.9 * self.ema_gain + 0.1 * sensory_gain
        
        relay_vec = vis.get("vis_vec", np.array([0.0, 0.0])) * self.ema_gain
        relay_features = vis.get("features", np.array([0.0, 0.0, 0.0, 0.0])) * self.ema_gain
        
        return {
            "relay_vec": relay_vec,
            "relay_features": relay_features,
            "touch": som.get("touch", 0.0),
            "pain": som.get("pain", 0.0)
        }


class PFC_Stripe:
    def __init__(self, id_):
        self.id = id_
        self.memory = np.array([0.0, 0.0])
        self.is_locked = False

    def update(self, input_vector: np.ndarray, gate_signal: float):
        if gate_signal > 0:
            self.memory = input_vector
            self.is_locked = False
        else:
            self.is_locked = True
        return self.memory


class BasalGanglia:
    def __init__(self):
        self.w_gate = np.random.rand(3)
        self.w_motor = np.ones(3)
        self.pvlv = PVLV_Learning()

    def select_action_and_gate(
        self,
        sensory_vec: np.ndarray,
        memory_vec: np.ndarray,
        drive_level: float,
        reward: float,
        learning_rate_mod: float,
        memory_gain: float = 1.0  # <--- NEW PARAMETER
    ):
        sensory_mag = float(np.linalg.norm(sensory_vec))
        dopamine = self.pvlv.step(sensory_mag, reward)

        # Apply the manual slider gain to the memory vector
        effective_memory = memory_vec * memory_gain 

        context = np.array([np.linalg.norm(sensory_vec), np.linalg.norm(effective_memory), drive_level], dtype=float)

        gate_activation = float(np.dot(self.w_gate, context) + dopamine)
        gate_signal = 1.0 if gate_activation > 0.5 else -1.0

        w_sensory = self.w_motor[0] * (1.0 + dopamine)
        w_memory = self.w_motor[1] * (1.0 - dopamine)
        
        # Use effective_memory here
        motor_out = (sensory_vec * w_sensory) + (effective_memory * w_memory)

        if learning_rate_mod > 0.1:
            lr = 0.01 * learning_rate_mod
            self.w_gate += lr * dopamine * context
            self.w_motor += lr * dopamine

            self.w_gate = np.clip(self.w_gate, -1.0, 1.0)
            self.w_motor = np.clip(self.w_motor, 0.0, 5.0)

        return motor_out, gate_signal, dopamine


class TimeCellPopulation:
    def __init__(self, n_cells=20):
        self.n_cells = n_cells
        self.cells = np.zeros(n_cells)
        self.peaks = np.linspace(0, 100, n_cells)
        self.sigmas = np.linspace(5, 20, n_cells)
        self.timer = 0
        self.active = False

    def reset(self):
        self.timer = 0
        self.active = True
        self.cells.fill(0.0)

    def step(self):
        if not self.active:
            return np.zeros(self.n_cells)

        self.timer += 1
        for i in range(self.n_cells):
            self.cells[i] = np.exp(-((self.timer - self.peaks[i]) ** 2) / (2 * self.sigmas[i] ** 2))
        return self.cells


class DendriticCluster:
    def __init__(self):
        self.pfc = PFC_Stripe(0)
        self.basal_ganglia = BasalGanglia()

        self.soma = MicrotubuleSimulator2D("Executive Soma", length_points=32)
        self.astrocyte = Astrocyte()
        self.trn = TRNGate()
        self.time_cells = TimeCellPopulation()

        self.vision_cortex = VisionCortex()
        self.somato_cortex = SomatoCortex()
        self.thalamus = Thalamus()

        # STC variables
        self.synaptic_tags = np.zeros(3)
        self.prp_level = 0.0

        # Theta / Grid / HD
        self.mt_theta = MicrotubuleSimulator2D("Theta Memory", length_points=32)
        self.mt_theta.H_BAR = 6.0
        self.grid_k = [
            np.array([np.cos(0), np.sin(0)]),
            np.array([np.cos(np.pi / 3), np.sin(np.pi / 3)]),
            np.array([np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)]),
        ]
        self.grid_phases = np.array([0.0, 0.0, 0.0])
        self.grid_scale = 0.8

        self.hd_cell_count = 36
        self.hd_cells = self._create_activity_bump(self.hd_cell_count, 0)
        self.hd_integration_weight = 2.0

        self.place_memories: List[np.ndarray] = []

        # Evolution weights (used below)
        self.weights = {"amygdala": 1.0, "striatum": 1.0, "hippocampus": 1.0}
        self.load_latest_memory()

    def _create_activity_bump(self, size, position):
        x = np.arange(size)
        bump = np.exp(-0.5 * np.minimum((x - position) ** 2, (size - np.abs(x - position)) ** 2) / (size / 10) ** 2)
        return bump / np.sum(bump)

    def update_hd_cells(self, angular_velocity):
        shift = -angular_velocity * self.hd_integration_weight * self.hd_cell_count / (2 * np.pi)
        f_hd = np.fft.fft(self.hd_cells)
        freqs = np.fft.fftfreq(self.hd_cell_count)
        f_hd *= np.exp(-1j * 2 * np.pi * freqs * shift)
        self.hd_cells = np.real(np.fft.ifft(f_hd))
        self.hd_cells += 0.01 * self._create_activity_bump(self.hd_cell_count, np.argmax(self.hd_cells))
        self.hd_cells /= np.sum(self.hd_cells)

    def get_head_direction(self):
        return np.argmax(self.hd_cells) * (2 * np.pi / self.hd_cell_count)

    def update_grid_cells(self, velocity, head_direction):
        rotation_matrix = np.array(
            [[np.cos(head_direction), -np.sin(head_direction)], [np.sin(head_direction), np.cos(head_direction)]]
        )
        for i in range(3):
            rotated_k = rotation_matrix @ self.grid_k[i]
            proj = float(np.dot(velocity, rotated_k))
            drift = float(np.random.normal(0, 0.001))
            self.grid_phases[i] += (proj * self.grid_scale) + drift
            self.grid_phases[i] %= (2 * np.pi)

    def get_grid_visualization(self):
        size = 32
        x = np.linspace(-np.pi, np.pi, size)
        y = np.linspace(-np.pi, np.pi, size)
        X, Y = np.meshgrid(x, y)
        W1 = np.cos(X * np.cos(0) + Y * np.sin(0) + self.grid_phases[0])
        W2 = np.cos(X * np.cos(np.pi / 3) + Y * np.sin(np.pi / 3) + self.grid_phases[1])
        W3 = np.cos(X * np.cos(2 * np.pi / 3) + Y * np.sin(2 * np.pi / 3) + self.grid_phases[2])
        return (W1 + W2 + W3 + 3) / 6.0

    def load_latest_memory(self):
        if os.path.exists("evolution_log.csv"):
            try:
                with open("evolution_log.csv", "r") as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        last_line = lines[-1].strip().split(",")
                        self.weights["amygdala"] = float(last_line[2])
                        self.weights["striatum"] = float(last_line[3])
                        self.weights["hippocampus"] = float(last_line[4])
            except Exception:
                pass

    def save_memory(self):
        with open("brain_genome.json", "w") as f:
            json.dump(self.weights, f)

    def update_stc(self, activity_vector, reward_signal, dt=0.05):
        self.synaptic_tags *= 0.95
        self.synaptic_tags += 0.1 * np.abs(activity_vector)
        self.synaptic_tags = np.clip(self.synaptic_tags, 0.0, 1.0)

        if reward_signal > 0.5:
            self.prp_level = 1.0
        else:
            self.prp_level *= 0.98

        consolidation = self.synaptic_tags * self.prp_level
        return consolidation

    def process_votes(
        self,
        frustration,
        dopamine,
        rat_vel,
        reward_signal,
        danger_level,
        pheromones_len,
        head_direction,
        vision_data,
        anesthetic_level=0.0,
        # --- NEW PARAMETERS ---
        fear_gain=1.0,
        memory_gain=1.0,
        energy_constraint=True,
        # NEW OPTIONAL SENSORY INPUTS (backwards compatible)
        whisker_hits=None,
        collision=False,
        sensory_blend=0.0,
        enable_sensory_cortex=False,
        enable_thalamus=False,
        extra_arousal=0.0,
    ):
        # ---- Apply genome weights so evolution actually matters ----
        w_amyg = clamp(self.weights.get("amygdala", 1.0), 0.1, 10.0)
        w_str  = clamp(self.weights.get("striatum", 1.0), 0.1, 10.0)
        w_hip  = clamp(self.weights.get("hippocampus", 1.0), 0.1, 10.0)

        self.update_grid_cells(rat_vel, head_direction)

        neural_activity = float(np.linalg.norm(rat_vel) + frustration)
        
        # --- LOGIC UPDATE: Energy Constraint Switch ---
        real_atp, ext_lactate = self.astrocyte.step(neural_activity)
        if energy_constraint:
            atp_availability = real_atp
        else:
            atp_availability = 1.0 # Infinite energy mode
            
        arousal = (dopamine * 0.5) + (frustration * 0.5) + extra_arousal
        if atp_availability < 0.2:
            arousal = 0.0

        amygdala_drive = float(danger_level) * w_amyg * fear_gain
        pfc_drive = float(frustration) * w_hip

        trn_modes = self.trn.step(arousal, amygdala_drive, pfc_drive)

        gain_sensory = 1.0 if trn_modes[0] == "TONIC" else 0.05
        gain_memory = 1.0 if trn_modes[1] == "TONIC" else 0.05

        if reward_signal > 0:
            self.time_cells.reset()
        _temporal_context = self.time_cells.step() 

        # --- REFACTORED SENSORY PATHWAY ---

        # 1. Compute raw old sensory vector (pre-gain)
        sensory_vec_old_raw = np.array([0.0, 0.0])
        closest_dist = 999.0
        for ray in vision_data:
            if ray["type"] == 3: # Target
                if ray["dist"] < closest_dist:
                    closest_dist = ray["dist"]
                    sensory_vec_old_raw = np.array([np.cos(ray["angle"]), np.sin(ray["angle"])])
        
        norm_old = np.linalg.norm(sensory_vec_old_raw)
        if norm_old > 0:
            sensory_vec_old_raw /= norm_old

        # Gate the old pathway
        sensory_vec_old_gated = sensory_vec_old_raw * gain_sensory

        # 2. Compute new sensory vector (gated)
        sensory_vec_new_gated = np.array([0.0, 0.0])
        pain = 0.0
        touch = 0.0

        if enable_sensory_cortex:
            vis_data = self.vision_cortex.encode(vision_data)
            som_data = self.somato_cortex.encode(whisker_hits, collision)
            sensory_vec_new_raw = vis_data["vis_vec"]
            pain = som_data["pain"]
            touch = som_data["touch"]
            
            norm_new = np.linalg.norm(sensory_vec_new_raw)
            if norm_new > 0:
                sensory_vec_new_raw /= norm_new
            vis_data["vis_vec"] = sensory_vec_new_raw

            if enable_thalamus:
                thalamus_out = self.thalamus.relay(vis_data, som_data, topdown_attention=pfc_drive, relay_gain=gain_sensory)
                sensory_vec_new_gated = thalamus_out["relay_vec"]
            else:
                # If no thalamus, just use the cortex output, but still gate it
                sensory_vec_new_gated = sensory_vec_new_raw * gain_sensory
        
        # 3. Blend the two gated vectors
        k = clamp(float(sensory_blend), 0.0, 1.0)
        sensory_vec = ((1 - k) * sensory_vec_old_gated) + (k * sensory_vec_new_gated)

        current_memory = self.pfc.memory * gain_memory

        plasticity_mod = (ext_lactate / 0.5) * (0.5 + 0.5 * w_str)
        plasticity_mod = float(np.clip(plasticity_mod, 0.0, 5.0))

        # --- LOGIC UPDATE: Pass memory_gain ---
        final_vector, gate_signal, internal_dopamine = self.basal_ganglia.select_action_and_gate(
            sensory_vec, current_memory, frustration, reward_signal, plasticity_mod, memory_gain=memory_gain
        )
        
        # ... rest of the function remains the same ...
        activity_proxy = np.array([np.linalg.norm(sensory_vec), np.linalg.norm(current_memory), frustration])
        consolidation_vector = self.update_stc(activity_proxy, reward_signal)

        if float(np.max(consolidation_vector)) > 0.1:
            self.basal_ganglia.w_gate += 0.001 * consolidation_vector * internal_dopamine

        self.pfc.update(sensory_vec, gate_signal)

        angle = float(np.arctan2(final_vector[1], final_vector[0]))
        if angle < 0:
            angle += 2 * np.pi

        soma_input = np.zeros((self.soma.Ny, self.soma.Nx))
        target_idx = int((angle / (2 * np.pi)) * self.soma.Nx) % 13
        soma_input[:, target_idx] += 5.0

        pump_map = soma_input * atp_availability
        d_soma, _ = self.soma.step(pump_map, lfp_signal=0.0, anesthetic_conc=anesthetic_level)

        theta_pump = (0.5 + (pheromones_len / 200.0)) * atp_availability
        d_theta, _ = self.mt_theta.step(theta_pump, lfp_signal=0.0, anesthetic_conc=anesthetic_level)

        return d_soma, d_theta, self.get_grid_visualization(), final_vector, self.astrocyte.glycogen, self.astrocyte.neuronal_atp, pain, touch



class Predator:
    def __init__(self, pos):
        self.pos = np.array(pos)
        self.vel = np.array([0.0, 0.0])
        self.speed = 0.35

    def hunt(self, rat_pos, grid, map_size):
        diff = rat_pos - self.pos
        dist = np.linalg.norm(diff)
        if dist > 0:
            desired_vel = (diff / dist) * self.speed
            next_pos = self.pos + desired_vel
            nx, ny = int(next_pos[0]), int(next_pos[1])
            if 0 <= nx < map_size and 0 <= ny < map_size:
                if grid[ny, nx] == 0:
                    self.pos = next_pos
                else:
                    if grid[ny, int(self.pos[0])] == 0:
                        self.pos[1] = next_pos[1]
                    elif grid[int(self.pos[1]), nx] == 0:
                        self.pos[0] = next_pos[0]


class Cerebellum:
    def __init__(self):
        self.predicted_pos = np.array([0.0, 0.0])
        self.correction_vector = np.array([0.0, 0.0])
        self.learning_rate = 0.1

    def predict(self, current_pos, efference_copy):
        self.predicted_pos = current_pos + (efference_copy * 0.15)

    def update(self, actual_pos):
        error = actual_pos - self.predicted_pos
        self.correction_vector = self.correction_vector * (1 - self.learning_rate) + error * self.learning_rate


class SimulationState:
    def __init__(self):
        self.brain = DendriticCluster()

        self.map_size = 40
        self.rat_pos = np.array([3.0, 3.0])
        self.rat_vel = np.array([0.0, 0.6])
        self.rat_heading = np.pi / 2  # match initial velocity

        self.targets: List[np.ndarray] = []
        self.predator = Predator([35.0, 3.0])
        self.cerebellum = Cerebellum()

        self.score = 0
        self.deaths = 0
        self.frustration = 0.0
        self.dopamine = 0.2
        self.serotonin = 0.5

        self.state = "AWAKE"
        self.dream_scans = []
        self.phantom_trace = []
        self.pheromones: List[List[float]] = []

        self.dream_timer = 0
        self.panic = 0

        # Add config defaults + helpful knobs
        self.config = {
            "speed": 1,
            "sensitivity": 0.05,
            "anesthetic": 0.0,
            "downsample": 1,
            "max_speed": 10,
            # --- NEW KNOBS ---
            "cerebellum_gain": 1.0,  # 0.0 = Clumsy, 1.0 = Precise
            "memory_gain": 1.0,      # 0.0 = Pure Sensory, 1.0 = Balanced
            "fear_gain": 1.0,        # 0.0 = Fearless
            "energy_constraint": 1.0, # 1.0 = Starvation possible, 0.0 = Infinite Energy
            "enable_sensory_cortex": 0.0,   # 0/1
            "enable_thalamus": 0.0,         # 0/1
            "sensory_blend": 0.0,           # 0..1
        }

        self.generation = 1
        self.best_fitness = 0.0
        self.frames_alive = 0
        self.run_history: List[float] = []
        self.genes_being_tested = copy.deepcopy(self.brain.weights)
        self.stable_genome = copy.deepcopy(self.brain.weights)

        # Whiskers / vision
        self.whisk_angle = 0.0
        self.whisk_phase = 0.0
        self.whisk_freq = 0.8
        self.whisk_amp = np.pi / 4.0
        self.whisker_hits = [False, False]
        self.vision_buffer: List[Dict[str, Any]] = []
        self.last_collision = False
        self.last_touch = 0.0

        if not os.path.exists("evolution_log.csv"):
            with open("evolution_log.csv", "w") as f:
                f.write("Generation,Fitness,Amygdala,Striatum,Hippocampus\n")
        else:
            try:
                with open("evolution_log.csv", "r") as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        self.generation = int(lines[-1].split(",")[0]) + 1
            except Exception:
                pass

        self.occupancy_grid = np.zeros((self.map_size, self.map_size), dtype=int)
        self.generate_map()

    def generate_map(self):
        while True:
            self.occupancy_grid = np.random.choice([0, 1], size=(self.map_size, self.map_size), p=[0.6, 0.4])
            for _ in range(5):
                new_grid = self.occupancy_grid.copy()
                for y in range(1, self.map_size - 1):
                    for x in range(1, self.map_size - 1):
                        neighbors = int(np.sum(self.occupancy_grid[y - 1 : y + 2, x - 1 : x + 2]))
                        if neighbors > 4:
                            new_grid[y, x] = 1
                        elif neighbors < 4:
                            new_grid[y, x] = 0
                self.occupancy_grid = new_grid

            self.occupancy_grid[0, :] = 1
            self.occupancy_grid[-1, :] = 1
            self.occupancy_grid[:, 0] = 1
            self.occupancy_grid[:, -1] = 1

            self.occupancy_grid[1:6, 1:6] = 0
            self.occupancy_grid[34:39, 34:39] = 0
            self.occupancy_grid[1:6, 34:39] = 0

            # BFS reachability
            test_grid = self.occupancy_grid.copy()
            start = (3, 3)
            queue = [start]
            test_grid[start] = 2
            reachable_tiles = [start]

            while queue:
                cx, cy = queue.pop(0)
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                        if test_grid[ny, nx] == 0:
                            test_grid[ny, nx] = 2
                            queue.append((nx, ny))
                            reachable_tiles.append((nx, ny))

            if len(reachable_tiles) > 200:
                self.rat_pos = np.array([3.0, 3.0])
                reachable_tiles.sort(key=lambda p: (p[0] - 3) ** 2 + (p[1] - 3) ** 2)

                self.targets = [self._spawn_single_target() for _ in range(3)]

                far_tiles = [t for t in reachable_tiles if (t[0] - 3) ** 2 + (t[1] - 3) ** 2 > 200]
                if far_tiles:
                    p_spawn = far_tiles[np.random.randint(len(far_tiles))]
                    self.predator = Predator([float(p_spawn[0]), float(p_spawn[1])])
                    self.pheromones = []
                    self.brain.astrocyte = Astrocyte()
                    break

    def check_collision(self, pos):
        def is_wall(x, y):
            ix, iy = int(x), int(y)
            if ix < 0 or ix >= self.map_size or iy < 0 or iy >= self.map_size:
                return True
            return self.occupancy_grid[iy, ix] == 1

        if is_wall(pos[0], pos[1]):
            return True

        radius = 0.3
        if is_wall(pos[0] + radius, pos[1]):
            return True
        if is_wall(pos[0] - radius, pos[1]):
            return True
        if is_wall(pos[0], pos[1] + radius):
            return True
        if is_wall(pos[0], pos[1] - radius):
            return True

        return False

    def _spawn_single_target(self):
        while True:
            tx, ty = np.random.randint(1, self.map_size - 1), np.random.randint(1, self.map_size - 1)
            if self.occupancy_grid[ty, tx] != 0:
                continue

            # BFS reachability
            test_grid = self.occupancy_grid.copy()
            start = (int(self.rat_pos[0]), int(self.rat_pos[1]))
            queue = [start]
            test_grid[start[1], start[0]] = 2

            reachable = False
            steps = 0
            max_steps = self.map_size * self.map_size

            while queue and steps < max_steps:
                cx, cy = queue.pop(0)
                if cx == tx and cy == ty:
                    reachable = True
                    break
                steps += 1
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                        if test_grid[ny, nx] == 0:
                            test_grid[ny, nx] = 2
                            queue.append((nx, ny))

            if reachable:
                return np.array([float(tx), float(ty)])

    def cast_ray(self, angle: float):
        dx, dy = np.cos(angle), np.sin(angle)

        min_dist = 15.0
        obj_type = 0

        # walls first
        for d in np.linspace(0, 15.0, 30):
            cx = self.rat_pos[0] + dx * d
            cy = self.rat_pos[1] + dy * d
            if self.check_collision([cx, cy]):
                min_dist = float(d)
                obj_type = 1
                break

        # predator (angle wrap aware)
        to_predator = self.predator.pos - self.rat_pos
        dist_predator = float(np.linalg.norm(to_predator))
        if dist_predator > 0:
            pred_angle = float(np.arctan2(to_predator[1], to_predator[0]))
            if angdiff(angle, pred_angle) < 0.1:
                if dist_predator < min_dist:
                    min_dist = dist_predator
                    obj_type = 2

        # targets (angle wrap aware)
        for target in self.targets:
            to_target = target - self.rat_pos
            dist_target = float(np.linalg.norm(to_target))
            if dist_target > 0:
                target_angle = float(np.arctan2(to_target[1], to_target[0]))
                if angdiff(angle, target_angle) < 0.1:
                    if dist_target < min_dist:
                        min_dist = dist_target
                        obj_type = 3

        return min_dist, obj_type

    def process_vision(self):
        self.vision_buffer = []
        heading = float(np.arctan2(self.rat_vel[1], self.rat_vel[0]))
        fov = np.pi * (2 / 3)

        for i in range(12):
            angle_offset = (i / 11.0 - 0.5) * fov
            ray_angle = heading + angle_offset
            dist, obj_type = self.cast_ray(ray_angle)
            self.vision_buffer.append({"dist": float(dist), "type": int(obj_type), "angle": float(ray_angle)})

    def update_whiskers(self):
        self.whisk_phase += self.whisk_freq
        self.whisk_angle = float(np.sin(self.whisk_phase) * self.whisk_amp)

        angles = [(-np.pi / 4) + self.whisk_angle, (np.pi / 4) - self.whisk_angle]

        heading = float(np.arctan2(self.rat_vel[1], self.rat_vel[0]))
        self.whisker_hits = [False, False]
        whisker_length = 5.0

        for i, angle in enumerate(angles):
            global_angle = heading + angle
            tip_x = self.rat_pos[0] + np.cos(global_angle) * whisker_length
            tip_y = self.rat_pos[1] + np.sin(global_angle) * whisker_length

            mid_x = self.rat_pos[0] + np.cos(global_angle) * (whisker_length * 0.5)
            mid_y = self.rat_pos[1] + np.sin(global_angle) * (whisker_length * 0.5)

            if self.check_collision([mid_x, mid_y]) or self.check_collision([tip_x, tip_y]):
                self.whisker_hits[i] = True

    def die(self):
        fitness = (self.score * 100) + (self.frames_alive * 0.01)
        self.run_history.append(float(fitness))
        BATCH_SIZE = 5

        with open("evolution_log.csv", "a") as f:
            f.write(
                f"{self.generation},{fitness:.2f},{self.brain.weights['amygdala']:.3f},{self.brain.weights['striatum']:.3f},{self.brain.weights['hippocampus']:.3f}\n"
            )

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
                change = 1.0 + float(np.random.normal(0, mutation_rate))
                self.brain.weights[k] = max(0.1, min(10.0, self.brain.weights[k] * change))

            self.genes_being_tested = copy.deepcopy(self.brain.weights)
            self.run_history = []
            self.generation += 1

        self.deaths += 1
        self.frames_alive = 0
        self.rat_pos = np.array([3.0, 3.0])
        self.rat_vel = np.array([0.0, 0.6])
        self.rat_heading = np.pi / 2
        self.pheromones = []
        self.frustration = 0.0
        self.panic = 0.0

        # reset short-term state
        self.brain.astrocyte = Astrocyte()
        self.brain.trn = TRNGate()
        self.brain.basal_ganglia = BasalGanglia()
        self.brain.place_memories = []

        while True:
            tx, ty = np.random.randint(1, 39), np.random.randint(1, 39)
            if self.occupancy_grid[ty, tx] == 0:
                dist = (tx - 3) ** 2 + (ty - 3) ** 2
                if dist > 200:
                    self.predator = Predator([float(tx), float(ty)])
                    break


sim = SimulationState()


@app.route("/")
def index():
    return render_template("index.html")


def serialize_state(
    soma_density: np.ndarray,
    theta_density: np.ndarray,
    grid: np.ndarray,
    downsample: int,
) -> Dict[str, Any]:
    soma_out = maybe_downsample(soma_density, downsample)
    theta_out = maybe_downsample(theta_density, downsample)
    grid_out = maybe_downsample(grid, downsample)

    return {
        "soma": soma_out.tolist(),
        "theta": theta_out.tolist() if isinstance(theta_out, np.ndarray) else float(theta_out),
        "grid": grid_out.tolist(),
        "hd": maybe_downsample(sim.brain.hd_cells, 1).tolist(),
    }


@app.route("/step", methods=["POST"])
def step():
    with sim_lock:
        req_data = request.json or {}

        # Clamp iterations
        requested = req_data.get("batch_size", sim.config["speed"])
        try:
            iterations = int(requested)
        except Exception:
            iterations = int(sim.config["speed"])

        max_speed = int(sim.config.get("max_speed", 10))
        iterations = int(np.clip(iterations, 1, max_speed))

        anesthetic_level = clamp(float(sim.config.get("anesthetic", 0.0)), 0.0, 1.0)
        downsample = int(sim.config.get("downsample", 1))
        downsample = int(np.clip(downsample, 1, 8))

        # Default placeholders so we can always return on last frame
        soma_density = np.zeros((sim.brain.soma.Ny, sim.brain.soma.Nx), dtype=float)
        d_theta = np.zeros((sim.brain.mt_theta.Ny, sim.brain.mt_theta.Nx), dtype=float)
        d_grid = np.zeros((32, 32), dtype=float)
        glycogen_level = float(sim.brain.astrocyte.glycogen)
        atp_level = float(sim.brain.astrocyte.neuronal_atp)

        for i in range(iterations):
            sim.update_whiskers()
            sim.process_vision()
            sim.frames_alive += 1

            current_atp = float(sim.brain.astrocyte.neuronal_atp)
            dist_cat = float(np.linalg.norm(sim.rat_pos - sim.predator.pos))
            
            # Use last_touch for arousal calculation
            arousal_from_touch = sim.last_touch * 0.1

            if current_atp <= 0.01 or dist_cat < 1.0:
                sim.die()
                return jsonify(
                    {
                        "rat": sim.rat_pos.tolist(),
                        "targets": [t.tolist() for t in sim.targets],
                        "predator": sim.predator.pos.tolist(),
                        "brain": {},
                        "stats": {
                            "frustration": 0,
                            "score": sim.score,
                            "status": "DEAD",
                            "energy": 0,
                            "atp": 0,
                            "deaths": sim.deaths,
                            "generation": sim.generation,
                        },
                        "reset_visuals": True,
                        "whiskers": {"angle": sim.whisk_angle, "hits": sim.whisker_hits},
                        "vision": sim.vision_buffer,
                    }
                )

            sim.dopamine = max(0.1, sim.dopamine * 0.995)
            if sim.frustration < 0.1:
                sim.serotonin = min(1.0, sim.serotonin + 0.005)

            # Head direction update needs to happen before TRN modes are checked
            new_heading = float(np.arctan2(sim.rat_vel[1], sim.rat_vel[0]))
            angular_velocity = new_heading - sim.rat_heading
            if angular_velocity > np.pi:
                angular_velocity -= 2 * np.pi
            if angular_velocity < -np.pi:
                angular_velocity += 2 * np.pi
            sim.rat_heading = new_heading

            sim.brain.update_hd_cells(angular_velocity)
            head_direction = sim.brain.get_head_direction()

            trn_modes = sim.brain.trn.modes

            # If both in burst: return microsleep on final iteration
            if trn_modes[0] == "BURST" and trn_modes[1] == "BURST":
                if i == iterations - 1:
                    soma_density, d_theta, d_grid, final_decision, glycogen_level, atp_level, _, _ = sim.brain.process_votes(
                        sim.frustration,
                        sim.dopamine,
                        sim.rat_vel,
                        0,
                        0,
                        len(sim.pheromones),
                        head_direction,
                        sim.vision_buffer,
                        anesthetic_level=anesthetic_level,
                        extra_arousal=arousal_from_touch,
                    )
                    return jsonify(
                        {
                            "rat": sim.rat_pos.tolist(),
                            "targets": [t.tolist() for t in sim.targets],
                            "predator": sim.predator.pos.tolist(),
                            "brain": serialize_state(soma_density, d_theta, d_grid, downsample),
                            "cerebellum": {
                                "predicted_pos": sim.cerebellum.predicted_pos.tolist(),
                                "correction_vector": sim.cerebellum.correction_vector.tolist(),
                            },
                            "stats": {
                                "frustration": sim.frustration,
                                "score": sim.score,
                                "status": "MICROSLEEP",
                                "dopamine": sim.dopamine,
                                "serotonin": sim.serotonin,
                                "energy": float(glycogen_level),
                                "atp": float(atp_level),
                                "deaths": sim.deaths,
                                "generation": sim.generation,
                            },
                            "phantom": sim.phantom_trace,
                            "pheromones": sim.pheromones,
                            "whiskers": {"angle": sim.whisk_angle, "hits": sim.whisker_hits},
                            "vision": sim.vision_buffer,
                        }
                    )
                continue

            danger_level = 0.0
            if dist_cat < 10.0:
                danger_level = 1.0 - (dist_cat / 10.0)

            reward_signal = 0.0
            for idx, target_pos in enumerate(sim.targets):
                if float(np.linalg.norm(sim.rat_pos - target_pos)) < 1.0:
                    sim.score += 1
                    sim.frustration = 0.0
                    sim.dopamine = 1.0
                    sim.brain.astrocyte.glycogen = 1.0
                    reward_signal = 1.0
                    sim.targets[idx] = sim._spawn_single_target()
                    break
            
            # NOTE: head direction is now computed above the microsleep check

            # Extract config settings
            cerebellum_gain = float(sim.config.get("cerebellum_gain", 1.0))
            memory_gain = float(sim.config.get("memory_gain", 1.0))
            fear_gain = float(sim.config.get("fear_gain", 1.0))
            use_energy = float(sim.config.get("energy_constraint", 1.0)) > 0.5
            
            collision_flag = sim.last_collision

            sensory_blend = float(sim.config.get("sensory_blend", 0.0))
            enable_sensory_cortex = float(sim.config.get("enable_sensory_cortex", 0.0)) > 0.5
            enable_thalamus = float(sim.config.get("enable_thalamus", 0.0)) > 0.5

            soma_density, d_theta, d_grid, final_decision, glycogen_level, atp_level, pain, touch = sim.brain.process_votes(
                sim.frustration,
                sim.dopamine,
                sim.rat_vel,
                reward_signal,
                danger_level,
                len(sim.pheromones),
                head_direction,
                sim.vision_buffer,
                anesthetic_level=anesthetic_level,
                # Pass new knobs
                fear_gain=fear_gain,
                memory_gain=memory_gain,
                energy_constraint=use_energy,
                whisker_hits=sim.whisker_hits,
                collision=collision_flag,
                sensory_blend=sensory_blend,
                enable_sensory_cortex=enable_sensory_cortex,
                enable_thalamus=enable_thalamus,
                extra_arousal=arousal_from_touch,
            )
            sim.last_touch = touch # Update for next step

            sim.cerebellum.predict(sim.rat_pos, final_decision)
            
            # --- APPLY CEREBELLUM GAIN ---
            final_decision_corrected = final_decision + (sim.cerebellum.correction_vector * cerebellum_gain)

            sim.rat_vel = (sim.rat_vel * 0.85) + (final_decision_corrected * 0.15)
            sim.rat_vel = np.nan_to_num(sim.rat_vel)

            tremor = np.random.randn(2) * 0.02
            sim.rat_vel += tremor

            s = float(np.linalg.norm(sim.rat_vel))
            max_speed = 0.3 + (sim.dopamine * 0.2)
            if s > max_speed:
                sim.rat_vel = (sim.rat_vel / (s + 1e-9)) * max_speed

            new_pos = sim.rat_pos + sim.rat_vel

            hit = sim.check_collision(new_pos)
            if hit:
                sim.rat_vel *= -0.5
                sim.frustration = min(1.0, sim.frustration + 0.1)
                sim.serotonin = max(0.1, sim.serotonin - 0.1)
            else:
                sim.rat_pos = new_pos
                sim.frustration = max(0.0, sim.frustration - 0.005)
            sim.last_collision = hit # Update for next tick

            # Pain increases frustration
            sim.frustration = min(1.0, sim.frustration + pain * 0.05)

            sim.cerebellum.update(sim.rat_pos)

            if np.random.rand() < 0.2:
                sim.pheromones.append(sim.rat_pos.tolist())
                if len(sim.pheromones) > 1000:
                    sim.pheromones.pop(0)

        # Return last-frame state
        return jsonify(
            {
                "rat": sim.rat_pos.tolist(),
                "targets": [t.tolist() for t in sim.targets],
                "predator": sim.predator.pos.tolist(),
                "brain": serialize_state(soma_density, d_theta, d_grid, downsample),
                "cerebellum": {
                    "predicted_pos": sim.cerebellum.predicted_pos.tolist(),
                    "correction_vector": sim.cerebellum.correction_vector.tolist(),
                },
                "stats": {
                    "frustration": sim.frustration,
                    "score": sim.score,
                    "status": "AWAKE" if "TONIC" in sim.brain.trn.modes else "MICROSLEEP",
                    "dopamine": sim.dopamine,
                    "serotonin": sim.serotonin,
                    "energy": float(glycogen_level),
                    "atp": float(atp_level),
                    "deaths": sim.deaths,
                    "generation": sim.generation,
                },
                "phantom": sim.phantom_trace,
                "pheromones": sim.pheromones,
                "whiskers": {"angle": sim.whisk_angle, "hits": sim.whisker_hits},
                "vision": sim.vision_buffer,
            }
        )


@app.route("/history", methods=["GET"])
def history():
    with sim_lock:
        data = []
        if os.path.exists("evolution_log.csv"):
            with open("evolution_log.csv", "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(row)
        return jsonify(data)


@app.route("/load_generation", methods=["POST"])
def load_generation():
    with sim_lock:
        payload = request.json or {}
        gen_id = str(payload.get("generation"))

        if os.path.exists("evolution_log.csv"):
            with open("evolution_log.csv", "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["Generation"] == gen_id:
                        sim.brain.weights["amygdala"] = float(row["Amygdala"])
                        sim.brain.weights["striatum"] = float(row["Striatum"])
                        sim.brain.weights["hippocampus"] = float(row["Hippocampus"])

                        sim.generate_map()
                        sim.deaths = 0

                        walls = []
                        for y in range(sim.map_size):
                            for x in range(sim.map_size):
                                if sim.occupancy_grid[y, x] == 1:
                                    walls.append([x, y, 1, 1])

                        return jsonify({"status": "loaded", "gen": gen_id, "walls": walls})

        return jsonify({"status": "error", "msg": "Gen not found"}), 404


@app.route("/reset", methods=["POST"])
def reset():
    with sim_lock:
        sim.generate_map()
        sim.frustration = 0
        sim.dopamine = 0.2
        sim.serotonin = 0.5
        sim.deaths = 0
        sim.generation = 1

        with open("evolution_log.csv", "w") as f:
            f.write("Generation,Fitness,Amygdala,Striatum,Hippocampus\n")

        walls = []
        for y in range(sim.map_size):
            for x in range(sim.map_size):
                if sim.occupancy_grid[y, x] == 1:
                    walls.append([x, y, 1, 1])

        return jsonify({"status": "reset", "walls": walls})


@app.route("/config", methods=["POST"])
def config():
    with sim_lock:
        payload = request.json or {}

        # Validate & clamp
        if "speed" in payload:
            sim.config["speed"] = int(np.clip(int(payload["speed"]), 1, int(sim.config.get("max_speed", 10))))
        if "max_speed" in payload:
            sim.config["max_speed"] = int(np.clip(int(payload["max_speed"]), 1, 200))
            sim.config["speed"] = int(np.clip(int(sim.config["speed"]), 1, sim.config["max_speed"]))
        if "sensitivity" in payload:
            sim.config["sensitivity"] = clamp(float(payload["sensitivity"]), 0.001, 1.0)
        if "anesthetic" in payload:
            sim.config["anesthetic"] = clamp(float(payload["anesthetic"]), 0.0, 1.0)
        if "downsample" in payload:
            sim.config["downsample"] = int(np.clip(int(payload["downsample"]), 1, 8))

        # Add new checks
        if "cerebellum_gain" in payload:
            sim.config["cerebellum_gain"] = clamp(float(payload["cerebellum_gain"]), 0.0, 5.0)
        if "memory_gain" in payload:
            sim.config["memory_gain"] = clamp(float(payload["memory_gain"]), 0.0, 5.0)
        if "fear_gain" in payload:
            sim.config["fear_gain"] = clamp(float(payload["fear_gain"]), 0.0, 10.0)
        if "energy_constraint" in payload:
            sim.config["energy_constraint"] = clamp(float(payload["energy_constraint"]), 0.0, 1.0)

        if "enable_sensory_cortex" in payload:
            sim.config["enable_sensory_cortex"] = clamp(float(payload["enable_sensory_cortex"]), 0.0, 1.0)
        if "enable_thalamus" in payload:
            sim.config["enable_thalamus"] = clamp(float(payload["enable_thalamus"]), 0.0, 1.0)
        if "sensory_blend" in payload:
            sim.config["sensory_blend"] = clamp(float(payload["sensory_blend"]), 0.0, 1.0)

        return jsonify({"status": "updated", "config": sim.config})


if __name__ == "__main__":
    # For a single-user demo: avoid threading surprises
    app.run(debug=True, port=5000, threaded=False)
