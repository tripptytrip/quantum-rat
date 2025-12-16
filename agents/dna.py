import json
import hashlib
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class AgentDNA:
    agent_id: str
    version: str
    params: dict[str, float|int|str|bool]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @staticmethod
    def from_json(json_str: str) -> "AgentDNA":
        data = json.loads(json_str)
        return AgentDNA(**data)

    def fingerprint(self) -> str:
        canonical_json = self.to_json()
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

def generate_population(seed: int, n: int) -> list[AgentDNA]:
    """
    Generates a deterministic population of agents.
    NOTE: seed is currently unused, but kept for future compatibility.
    """
    population = []
    for i in range(n):
        exploration_bias = i / (n - 1) if n > 1 else 0.5
        pain_avoidance = 1.0 - exploration_bias
        turn_bias = (-1)**i * 0.1

        population.append(
            AgentDNA(
                agent_id=f"agent_{i:04d}",
                version="1.0.0",
                params={
                    "exploration_bias": exploration_bias,
                    "pain_avoidance": pain_avoidance,
                    "turn_bias": turn_bias,
                },
            )
        )
    return population