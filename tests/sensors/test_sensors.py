from brain.contracts import Action
from core.entities import Agent
from core.rng import spawn_streams
from core.sensors import gather_observation
from core.world import World


def test_gather_observation_includes_egomotion():
    streams = spawn_streams(seed=7, names=["world", "noise", "sensors_vision", "sensors_noise"])
    world = World()
    agent = Agent(id=1, pos=(0.0, 0.0))
    world.add_agent(agent)

    world.step(action=Action(name="FORWARD", thrust=0.5, turn=0.1))  # move agent to produce deltas
    obs = gather_observation(agent, vision_stream=streams["sensors_vision"], noise_stream=streams["sensors_noise"])

    assert obs.forward_delta != 0.0 or obs.turn_delta != 0.0
    obs.validate()
