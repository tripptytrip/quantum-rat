from brain.contracts import Action
from core.entities import Agent
from core.world import World


def test_world_step_deterministic_with_action():
    action = Action(name="FORWARD", thrust=1.0, turn=0.1)
    world_a = World()
    world_b = World()
    agent_a = Agent(id=1, pos=(0.0, 0.0))
    agent_b = Agent(id=1, pos=(0.0, 0.0))
    world_a.add_agent(agent_a)
    world_b.add_agent(agent_b)

    for _ in range(5):
        world_a.step(action=action)
        world_b.step(action=action)

    assert agent_a.pos == agent_b.pos
    assert agent_a.heading == agent_b.heading
