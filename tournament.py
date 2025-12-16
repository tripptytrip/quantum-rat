import numpy as np
from rat_agent import RatAgent
from predator import Predator

# Trait metadata for evolvable parameters. Ranges mirror the API clamps in app.py.
DNA_TRAITS = [
    # Core neuromodulators / learning
    {"name": "fear_gain", "type": "float", "min": 0.0, "max": 10.0, "init_range": (0.5, 3.0), "mutate_scale": 0.2},
    {"name": "memory_gain", "type": "float", "min": 0.0, "max": 5.0, "init_range": (0.5, 2.0), "mutate_scale": 0.2},
    {"name": "dopamine_tonic", "type": "float", "min": -0.3, "max": 0.8, "init_range": (0.1, 0.5)},
    {"name": "cerebellum_gain", "type": "float", "min": 0.0, "max": 5.0, "init_range": (0.5, 2.0)},
    {"name": "energy_constraint", "type": "float", "min": 0.0, "max": 1.0, "init_range": (0.2, 1.0)},
    # Sensory routing
    {"name": "enable_sensory_cortex", "type": "bool"},
    {"name": "enable_thalamus", "type": "bool"},
    {"name": "sensory_blend", "type": "float", "min": 0.0, "max": 1.0, "init_range": (0.2, 1.0)},
    # Replay / hippocampus
    {"name": "enable_replay", "type": "bool"},
    {"name": "replay_steps", "type": "int", "min": 0, "max": 50, "init_range": (5, 25)},
    {"name": "replay_len", "type": "int", "min": 10, "max": 5000, "init_range": (200, 1200)},
    {"name": "replay_strength", "type": "float", "min": 0.0, "max": 2.0, "init_range": (0.05, 0.5)},
    {"name": "replay_bridge_prob", "type": "float", "min": 0.0, "max": 1.0, "init_range": (0.1, 0.5)},
    # Stress / panic
    {"name": "panic_gain", "type": "float", "min": 0.0, "max": 5.0, "init_range": (0.5, 2.5)},
    {"name": "cortisol_gain", "type": "float", "min": 0.0, "max": 5.0, "init_range": (0.2, 1.5)},
    {"name": "cortisol_decay", "type": "float", "min": 0.0, "max": 0.1, "init_range": (0.001, 0.02)},
    {"name": "panic_trn_gain", "type": "float", "min": 0.0, "max": 5.0, "init_range": (0.25, 2.0)},
    {"name": "panic_motor_bias", "type": "float", "min": 0.0, "max": 5.0, "init_range": (0.25, 2.0)},
    {"name": "stress_learning_gain", "type": "float", "min": 0.0, "max": 5.0, "init_range": (0.25, 2.0)},
    # Predictive processing
    {"name": "enable_predictive_processing", "type": "bool"},
    {"name": "pp_lr", "type": "float", "min": 0.0, "max": 1.0, "init_range": (0.01, 0.1)},
    {"name": "pp_weight_decay", "type": "float", "min": 0.0, "max": 1.0, "init_range": (0.0001, 0.1)},
    {"name": "pp_error_tau", "type": "float", "min": 0.0, "max": 10.0, "init_range": (0.1, 2.0)},
    {"name": "pp_surprise_gain", "type": "float", "min": 0.0, "max": 5.0, "init_range": (0.5, 2.5)},
    {"name": "pp_explore_gain", "type": "float", "min": 0.0, "max": 5.0, "init_range": (0.25, 2.0)},
    {"name": "pp_error_scale", "type": "float", "min": 0.0, "max": 10.0, "init_range": (0.5, 3.0)},
    # Place cells / navigation
    {"name": "enable_place_cells", "type": "bool"},
    {"name": "place_n", "type": "int", "min": 16, "max": 1024, "init_range": (128, 512)},
    {"name": "place_sigma", "type": "float", "min": 0.5, "max": 5.0, "init_range": (1.0, 3.0)},
    {"name": "place_lr", "type": "float", "min": 0.0, "max": 0.5, "init_range": (0.001, 0.05)},
    {"name": "place_decay", "type": "float", "min": 0.0, "max": 0.1, "init_range": (0.0001, 0.01)},
    {"name": "place_goal_lr", "type": "float", "min": 0.0, "max": 0.5, "init_range": (0.001, 0.05)},
    {"name": "place_nav_gain", "type": "float", "min": 0.0, "max": 5.0, "init_range": (0.25, 2.5)},
    {"name": "place_nav_blend", "type": "float", "min": 0.0, "max": 1.0, "init_range": (0.1, 0.6)},
    {"name": "goal_reward_threshold", "type": "float", "min": 0.0, "max": 1.0, "init_range": (0.25, 0.75)},
    # Microtubules
    {"name": "mt_causal", "type": "bool"},
    {"name": "mt_mod_plasticity", "type": "float", "min": 0.0, "max": 2.0, "init_range": (0.25, 1.5)},
    {"name": "mt_mod_explore", "type": "float", "min": 0.0, "max": 2.0, "init_range": (0.1, 1.0)},
    {"name": "mt_mod_gate", "type": "float", "min": 0.0, "max": 2.0, "init_range": (0.05, 1.0)},
    {"name": "mt_readout_tau", "type": "float", "min": 0.0, "max": 5.0, "init_range": (0.1, 1.5)},
    {"name": "mt_neighbor_mode", "type": "choice", "choices": ["legacy", "canonical"], "mutate_prob": 0.25},
    # Working memory
    {"name": "wm_slots", "type": "int", "min": 1, "max": 10, "init_range": (3, 8)},
    {"name": "wm_write_threshold", "type": "float", "min": 0.0, "max": 1.0, "init_range": (0.1, 0.5)},
    {"name": "wm_decay_rate", "type": "float", "min": 0.0, "max": 1.0, "init_range": (0.005, 0.1)},
    {"name": "wm_motor_gain", "type": "float", "min": 0.0, "max": 5.0, "init_range": (0.5, 2.0)},
]


class TournamentManager:
    def __init__(self, sim, base_config, map_size=40):
        self.sim = sim
        self.base_config = base_config
        self.map_size = map_size
        self.agents = []
        self.predators = []
        self.round = 1
        self.max_rounds = 3
        self.active = False
        self.match_log = []
        self.rng = np.random.default_rng()

    def _sample_trait(self, trait):
        name = trait["name"]
        t_type = trait["type"]
        init_range = trait.get("init_range", (trait.get("min", 0), trait.get("max", 1)))

        if t_type == "float":
            lo, hi = init_range
            return float(np.clip(self.rng.uniform(lo, hi), trait["min"], trait["max"]))
        if t_type == "int":
            lo, hi = int(init_range[0]), int(init_range[1])
            return int(np.clip(self.rng.integers(lo, hi + 1), trait["min"], trait["max"]))
        if t_type == "bool":
            return bool(self.rng.random() < 0.5)
        if t_type == "choice":
            return self.rng.choice(trait["choices"]).item()
        return None

    def _mutate_trait(self, value, trait):
        t_type = trait["type"]
        mutate_scale = trait.get("mutate_scale", 0.15)

        if t_type == "float":
            factor = self.rng.uniform(1 - mutate_scale, 1 + mutate_scale)
            mutated = float(np.clip(value * factor, trait["min"], trait["max"]))
            return mutated
        if t_type == "int":
            factor = self.rng.uniform(1 - mutate_scale, 1 + mutate_scale)
            mutated = int(np.clip(round(value * factor), trait["min"], trait["max"]))
            return mutated
        if t_type == "bool":
            mutate_prob = trait.get("mutate_prob", 0.2)
            if self.rng.random() < mutate_prob:
                return not value
            return value
        if t_type == "choice":
            mutate_prob = trait.get("mutate_prob", 0.2)
            if self.rng.random() < mutate_prob:
                choices = [c for c in trait["choices"] if c != value]
                return self.rng.choice(choices).item()
            return value
        return value

    def _initial_trait_value(self, trait):
        """Seed initial DNA from current config when available, else sample."""
        name = trait["name"]
        base_val = self.base_config.get(name, None)
        if base_val is not None:
            t_type = trait["type"]
            if t_type == "float":
                return float(np.clip(self._mutate_trait(float(base_val), trait), trait["min"], trait["max"]))
            if t_type == "int":
                return int(np.clip(round(self._mutate_trait(int(base_val), trait)), trait["min"], trait["max"]))
            if t_type == "bool":
                return bool(base_val)
            if t_type == "choice" and base_val in trait["choices"]:
                return base_val
        return self._sample_trait(trait)

    def start_tournament(self):
        self.active = True
        self.round = 1
        self.match_log = []
        print("TOURNAMENT: Starting new Championship.")

        # Create predators outside the maze
        self.predators = [
            Predator([-20, self.map_size / 2], "vertical"),  # West
            Predator([self.map_size + 20, self.map_size / 2], "vertical"),  # East
            Predator([self.map_size / 2, -20], "horizontal"),  # North
            Predator([self.map_size / 2, self.map_size + 20], "horizontal"),  # South
        ]

        self.agents = [
            self._spawn_agent(0, "Rat A"),
            self._spawn_agent(1, "Rat B"),
            self._spawn_agent(2, "Rat C"),
            self._spawn_agent(3, "Rat D"),
        ]
        
        self._create_quartered_map_and_food()

    def _create_quartered_map_and_food(self):
        # Create a quartered map
        self.sim.occupancy_grid.fill(0)
        self.sim.occupancy_grid[self.map_size // 2, :] = 1
        self.sim.occupancy_grid[:, self.map_size // 2] = 1
        self.sim.occupancy_grid[self.map_size // 2, self.map_size // 2] = 0 # Hole in the middle

        # Spawn food in each quadrant
        self.sim.targets = []
        quadrants = [
            (0, self.map_size // 2, 0, self.map_size // 2),
            (self.map_size // 2, self.map_size, 0, self.map_size // 2),
            (0, self.map_size // 2, self.map_size // 2, self.map_size),
            (self.map_size // 2, self.map_size, self.map_size // 2, self.map_size),
        ]

        for x_start, x_end, y_start, y_end in quadrants:
            for _ in range(2):
                while True:
                    tx = self.rng.integers(x_start + 1, x_end - 1)
                    ty = self.rng.integers(y_start + 1, y_end - 1)
                    if self.sim.occupancy_grid[ty, tx] == 0:
                        self.sim.targets.append(np.array([float(tx), float(ty)]))
                        break

    def _spawn_agent(self, id, label, parent_dna=None):
        dna = {}
        if parent_dna:
            for trait in DNA_TRAITS:
                name = trait["name"]
                parent_val = parent_dna.get(name, self._initial_trait_value(trait))
                dna[name] = self._mutate_trait(parent_val, trait)
        else:
            for trait in DNA_TRAITS:
                dna[trait["name"]] = self._initial_trait_value(trait)

        # Start in center of each quadrant
        half = self.map_size / 2
        quarter = self.map_size / 4
        starts = [
            [quarter, quarter],
            [half + quarter, quarter],
            [quarter, half + quarter],
            [half + quarter, half + quarter],
        ]
        pos = starts[id % 4]

        agent = RatAgent(id, pos, self.base_config, dna=dna, rng_seed=np.random.randint(0, 1000))
        agent.label = label
        return agent

    def _check_targets(self, agent, targets):
        """Returns 1.0 if target eaten, else 0.0. Modifies targets list in-place."""
        reward = 0.0
        eaten_idx = -1

        for i, t in enumerate(targets):
            if np.linalg.norm(agent.pos - t) < 1.0:
                reward = agent.receive_reward()
                eaten_idx = i
                break

        if eaten_idx != -1:
            # Respawn target immediately
            while True:
                tx = self.rng.integers(1, self.map_size - 1)
                ty = self.rng.integers(1, self.map_size - 1)
                safe = True
                for a in self.agents:
                    if np.linalg.norm(a.pos - np.array([tx, ty])) < 5.0:
                        safe = False
                if not safe:
                    continue

                for p in self.predators:
                    if np.linalg.norm(p.pos - np.array([tx, ty])) < 10.0:
                        safe = False

                if safe:
                    targets[eaten_idx] = np.array([float(tx), float(ty)])
                    break

        return reward

    def step(self, dt, env_state):
        positions = []

        for agent in self.agents:
            if agent.deaths == 0:
                reward = self._check_targets(agent, env_state["targets"])

                for predator in self.predators:
                    dist = np.linalg.norm(agent.pos - predator.pos)
                    if dist < 1.0:
                        agent.deaths = 1
                        print(f"TOURNAMENT: Agent {agent.id} died.")
                        break # No need to check other predators
                
                if agent.deaths == 0:
                    agent.step(dt, env_state, external_reward=reward)

            positions.append(
                {
                    "id": agent.id,
                    "pos": agent.pos.tolist(),
                    "heading": agent.heading,
                    "alive": agent.deaths == 0,
                    "score": agent.score,
                    "label": getattr(agent, "label", f"Rat {agent.id}"),
                    # EXPOSE DNA FOR FRONTEND
                    "dna": agent.dna,
                    "round": self.round,
                }
            )

        return positions

    def end_round(self):
        if not self.agents:
            # Safety: nothing to score; end tournament
            self.active = False
            print("TOURNAMENT: No agents available; ending tournament.")
            return True

        self.agents.sort(key=lambda x: x.score, reverse=True)
        winner = self.agents[0]
        print(f"TOURNAMENT: Round {self.round} Winner: {winner.label} (Score: {winner.score})")

        if self.round < self.max_rounds:
            self.round += 1
            champ_dna = winner.dna
            # Evolve: Winner stays, 2 variants join
            self.agents = [
                self._spawn_agent(0, "Champion", champ_dna),
                self._spawn_agent(1, "Challenger A", champ_dna),
                self._spawn_agent(2, "Wildcard", None),  # Wildcard
            ]
            return False
        else:
            self.active = False
            return True
