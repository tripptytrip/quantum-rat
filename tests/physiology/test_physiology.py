from brain.contracts import Action
from core.entities import Agent
from core.physiology import Astrocyte
from core.world import World


def test_atp_throttle_clamps_motion():
    world_full = World()
    world_low = World()
    agent_full = Agent(id=1, pos=(0.0, 0.0))
    agent_low = Agent(id=1, pos=(0.0, 0.0))
    world_full.add_agent(agent_full)
    world_low.add_agent(agent_low)
    action = Action(name="FORWARD", thrust=1.0, turn=0.1)

    # energy_scale=1 vs 0 to demonstrate clamping with identical actions
    world_full.step(action=action, energy_scale=1.0)
    world_low.step(action=action, energy_scale=0.0)

    assert agent_full.pos != agent_low.pos
    assert agent_low.pos == (0.0, 0.0)


def test_astrocyte_depletes_atp_and_uses_glycogen():
    astro = Astrocyte(atp=0.1, glycogen=1.0, glycogen_to_atp_yield=0.5, atp_cost=0.2)
    throttle = astro.tick(demand=1.0)
    assert throttle < 1.0
    assert astro.glycogen < 1.0  # pulled from glycogen to cover deficit
