"""US-4 (first part): the user's gel config, attached to the image."""

import numpy as np
import pytest

from gelanalyzer.config import LADDERS, GelConfig, Ladder
from gelanalyzer.io import GelImage, load_gel
from scn_factory import write_fake_scn


def test_config_from_a_plain_dict():
    config = GelConfig.from_dict({"n_lanes": 15, "ladder_lanes": [4], "ladder": "Precision Plus Unstained",
                                  "gel_type": "4-20% gradient"})
    assert config.n_lanes == 15
    assert config.ladder_lanes == (4,)
    assert config.ladder.kda[0] == 250 and config.ladder.kda[-1] == 10
    assert config.gel_type == "4-20% gradient"


def test_precision_plus_unstained_is_known():
    ladder = LADDERS["Precision Plus Unstained"]
    assert ladder.kda == (250, 150, 100, 75, 50, 37, 25, 20, 15, 10)
    assert ladder.thick == (75, 50, 25)
    assert len(ladder) == 10


def test_a_custom_ladder_can_be_given_directly():
    config = GelConfig(n_lanes=10, ladder_lanes=(1,), ladder=Ladder("my ladder", (100, 50, 25)))
    assert config.ladder.kda == (100, 50, 25)


@pytest.mark.parametrize("bad", [dict(n_lanes=0), dict(n_lanes=10, ladder_lanes=(11,)),
                                 dict(n_lanes=10, ladder_lanes=(0,))])
def test_impossible_configs_are_refused(bad):
    with pytest.raises(ValueError):
        GelConfig(**bad)


def test_unknown_ladder_name_gives_a_helpful_error():
    with pytest.raises(ValueError, match="Precision Plus Unstained"):    # lists the known ones
        GelConfig.from_dict({"n_lanes": 10, "ladder_lanes": [1], "ladder": "nonsense"})


def test_load_gel_attaches_the_config(tmp_path):
    path = write_fake_scn(tmp_path / "gel.scn", np.zeros((20, 30), dtype=np.uint16))
    gel = load_gel(path, config={"n_lanes": 15, "ladder_lanes": [4]})
    assert gel.config.n_lanes == 15
    assert load_gel(path).config is None           # no config given: none attached


def test_inverted_keeps_the_config():
    gel = GelImage(pixels=np.array([[1.0, 2.0]]), config=GelConfig(n_lanes=2))
    assert gel.inverted().config == gel.config
