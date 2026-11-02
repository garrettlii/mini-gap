"""Phase 2 contracts: skills.  uv run pytest --phase 2

Add one test per skill as you implement it: 20 dev seeds, assert its post-check.
"""

import numpy as np
import pytest

import gap.skills as skills
from gap.skills.base import Skill

pytestmark = pytest.mark.phase(2)


@pytest.mark.parametrize("name", skills.names())
def test_every_skill_has_a_post_check(name):
    cls = skills.get(name)
    assert cls.post is not Skill.post, f"{name} must override post()"


def test_locate_object_gt_matches_state():
    from gap.sim.task_sampler import sample
    from gap.sim.workcell import Workcell
    cell, skill = Workcell(), skills.get("locate_object_gt")()
    for seed in range(50000, 50020):
        s = cell.reset(sample("place", seed=seed))
        res = skill.run(cell, {"color": "red"}, skill.resolve_params())
        assert res.ok
        assert np.linalg.norm(res.outputs["pose"].p - s["objects"]["red_cube"].p) < 1e-3
        assert skill.post(cell, {"color": "red"}, res.outputs, {})[0]
