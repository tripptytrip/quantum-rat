import numpy as np
from rat_agent import RatAgent


class TournamentManager:
    def __init__(self, base_config, map_size=40):
        self.base_config = base_config
        self.map_size = map_size
        self.agents = []
        self.round = 1
        self.max_rounds = 3
        self.active = False
        self.match_log = []
        self.rng = np.random.default_rng()

    def start_tournament(self):
        self.active = True
        self.round = 1
        self.match_log = []
        print("TOURNAMENT: Starting new Championship.")
        self.agents = [
            self._spawn_agent(0, "Random A"),
            self._spawn_agent(1, "Random B"),
            self._spawn_agent(2, "Random C"),
        ]

    def _spawn_agent(self, id, label, parent_dna=None):
        dna = {}
        if parent_dna:
            dna = parent_dna.copy()
            # Mutate: +/- 20%
            dna["fear_gain"] *= np.random.uniform(0.8, 1.2)
            dna["memory_gain"] *= np.random.uniform(0.8, 1.2)
        else:
            # Random DNA
            dna = {
                "fear_gain": np.random.uniform(0.5, 3.0),
                "memory_gain": np.random.uniform(0.5, 2.0),
                "place_nav_gain": np.random.uniform(0.5, 2.0),
            }

        # Start in corners (top-left, top-right, bottom-left)
        starts = [[3, 3], [self.map_size - 4, 3], [3, self.map_size - 4]]
        pos = starts[id % 3]

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
                if safe:
                    targets[eaten_idx] = np.array([float(tx), float(ty)])
                    break

        return reward

    def step(self, dt, env_state):
        positions = []

        for agent in self.agents:
            if agent.deaths == 0:
                reward = self._check_targets(agent, env_state["targets"])

                dist = np.linalg.norm(agent.pos - env_state["predator"])
                if dist < 1.0:
                    agent.deaths = 1
                    print(f"TOURNAMENT: Agent {agent.id} died.")
                else:
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
                self._spawn_agent(2, "Challenger B", None),  # Wildcard
            ]
            return False
        else:
            self.active = False
            return True
