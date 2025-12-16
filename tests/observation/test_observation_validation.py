import math

import pytest

from brain.contracts import Observation, VisionRay


def test_observation_validation_passes():
    obs = Observation(
        vision_rays=(VisionRay(dist=0.5, obj_type="wall", angle=0.0),),
        whisker_hits=(False, True),
        pain_signal=0.1,
        forward_delta=0.0,
        turn_delta=0.0,
    )
    obs.validate()  # should not raise


@pytest.mark.parametrize("pain", [-0.1, 1.1])
def test_observation_validation_rejects_out_of_range_pain(pain):
    obs = Observation(
        vision_rays=(VisionRay(dist=0.5, obj_type="", angle=0.0),),
        whisker_hits=(False, False),
        pain_signal=pain,
        forward_delta=0.0,
        turn_delta=0.0,
    )
    with pytest.raises(ValueError):
        obs.validate()
