from agents.dna import AgentDNA
from core.engine import Engine

class Agent:
    def __init__(self, dna: AgentDNA):
        self.dna = dna

    def configure_engine(self, engine: Engine) -> None:
        """
        Configures the engine with the agent's DNA.
        For now, this is a no-op, but the structure is here for future use.
        """
        # Example of what could be done:
        # if "exploration_bias" in self.dna.params:
        #     engine.some_module.exploration_bias = self.dna.params["exploration_bias"]
        pass