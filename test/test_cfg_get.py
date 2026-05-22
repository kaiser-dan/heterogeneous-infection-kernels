import pytest
from src.data_processing.utils import cfg_get


def test_cfg_get_top_level_wins():
    cfg = {"foo": 1, "model": {"foo": 2}}
    assert cfg_get(cfg, "foo") == 1


def test_cfg_get_falls_back_to_model_namespace():
    cfg = {"model": {"foo": 2}}
    assert cfg_get(cfg, "foo") == 2


def test_cfg_get_returns_default_when_missing():
    cfg = {"model": {}}
    assert cfg_get(cfg, "foo", default=42) == 42


def test_cfg_get_default_none_when_missing():
    assert cfg_get({}, "foo") is None


def test_cfg_get_tolerates_non_dict_model():
    cfg = {"model": "not-a-dict"}
    assert cfg_get(cfg, "foo", default="x") == "x"
