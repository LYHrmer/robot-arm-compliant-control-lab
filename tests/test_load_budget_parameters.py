import numpy as np
import pytest

from tools.load_aware_compensation import LoadAwareCompensation

KW = {"mode": "online", "max_force": 8.0, "minimum_force": 6.0, "load_margin": 0.25,
          "load_time_constant": 0.20}


def make(**overrides):
    kw = dict(KW)
    kw.update(overrides)
    return LoadAwareCompensation(**kw)


@pytest.mark.parametrize("prop, expected", [
    ("load_estimate_n", 0.0),
    ("next_budget_n", KW["minimum_force"]),
    ("applied_budget_n", KW["minimum_force"]),
    ("projected_load_n", 0.0),
    ("measurement_used_for_budget", False),
])
def test_initial_state_and_properties_are_readonly(prop, expected):
    c = make()
    value = getattr(c, prop)
    if isinstance(expected, bool):
        assert bool(value) is expected
    else:
        assert value == pytest.approx(expected)
    with pytest.raises(AttributeError):
        setattr(c, prop, expected)


@pytest.mark.parametrize("overrides", [
    {"mode": "offline"},
    {"mode": "ONLINE"},
    {"minimum_force": 0.0},
    {"minimum_force": -1.0},
    {"minimum_force": 8.5},
    {"minimum_force": float("nan")},
    {"load_margin": float("nan")},
    {"load_margin": float("inf")},
    {"load_margin": -0.01},
    {"load_time_constant": 0.0},
    {"load_time_constant": -0.1},
    {"load_time_constant": float("nan")},
])
def test_constructor_rejects_invalid_arguments(overrides):
    with pytest.raises(ValueError):
        make(**overrides)


@pytest.mark.parametrize("bad", [
    np.zeros(2),
    np.zeros(4),
    np.zeros((3, 1)),
    np.zeros((1, 3)),
    np.float64(1.0),
    np.array([1.0, np.nan, 3.0]),
    np.array([np.inf, 0.0, 0.0]),
    np.array([0.0, 0.0, -np.inf]),
])
def test_set_force_measurement_rejects_invalid_input(bad):
    c = make()
    with pytest.raises(ValueError):
        c.set_force_measurement(bad)
    assert c._pending_measurement is None


@pytest.mark.parametrize("force", [
    np.array([1.0, 2.0, 3.0]),
    np.array([-0.5, 0.0, 4.25]),
])
def test_set_force_measurement_stores_owned_copy(force):
    c = make()
    c.set_force_measurement(force)
    assert c._pending_measurement is not force
    original = force.copy()
    force[:] = 99.0
    np.testing.assert_allclose(c._pending_measurement, original)


@pytest.mark.parametrize("first, second", [
    (np.array([1.0, 0.0, 0.0]), np.array([0.0, 2.0, 0.0])),
    (np.array([3.0, 3.0, 3.0]), np.array([-1.0, -2.0, -3.0])),
])
def test_second_measurement_replaces_queued_one(first, second):
    c = make()
    c.set_force_measurement(first)
    c.set_force_measurement(second)
    np.testing.assert_allclose(c._pending_measurement, second)


@pytest.mark.parametrize("bad", [np.zeros(2), np.array([0.0, np.nan, 0.0])])
def test_rejected_measurement_clears_previous_one(bad):
    c = make()
    c.set_force_measurement(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        c.set_force_measurement(bad)
    assert c._pending_measurement is None


@pytest.mark.parametrize("force", [np.array([1.0, 2.0, 3.0]), np.zeros(3)])
def test_reset_restores_initial_state(force):
    c = make()
    c.set_force_measurement(force)
    c.reset()
    assert c._pending_measurement is None
    assert c.load_estimate_n == pytest.approx(0.0)
    assert c.next_budget_n == pytest.approx(KW["minimum_force"])
    assert c.applied_budget_n == pytest.approx(KW["minimum_force"])
    assert not c.measurement_used_for_budget


@pytest.mark.parametrize("minimum_force", [0.5, 3.5, 8.0])
def test_nondefault_minimum_initializes_budgets(minimum_force):
    c = make(minimum_force=minimum_force)
    assert c.next_budget_n == pytest.approx(minimum_force)
    assert c.applied_budget_n == pytest.approx(minimum_force)
