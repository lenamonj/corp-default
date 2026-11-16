import numpy as np
import pytest
from scipy import stats

from corp_default.survival import cumulative_incidence, kaplan_meier


def test_kaplan_meier_by_hand():
    km = kaplan_meier([1, 2, 2, 3, 4], [1, 0, 1, 1, 0]).set_index("time")["survival"]
    assert list(km.loc[[1, 2, 3, 4]]) == pytest.approx([0.8, 0.6, 0.3, 0.3])


def test_kaplan_meier_with_delayed_entry_by_hand():
    km = kaplan_meier([1, 2, 2, 3, 4], [1, 0, 1, 1, 0], entry=[0, 0, 0, 2, 0]).set_index("time")
    assert list(km.loc[[1, 2, 3, 4], "survival"]) == pytest.approx([0.75, 0.5, 0.25, 0.25])
    assert list(km.loc[[1, 2, 3, 4], "at_risk"]) == [4, 3, 2, 1]


def test_kaplan_meier_greenwood_by_hand():
    # S(1) = 0.8 with n = 5, d = 1: var = 0.8**2 * 1 / (5 * 4) = 0.032
    km = kaplan_meier([1, 2, 2, 3, 4], [1, 0, 1, 1, 0]).set_index("time")
    assert km.loc[1, "se"] == pytest.approx(np.sqrt(0.032))


def test_kaplan_meier_agrees_with_scipy_without_delayed_entry():
    rng = np.random.default_rng(42)
    duration = rng.integers(1, 21, size=400)
    event = rng.random(400) < 0.4
    km = kaplan_meier(duration, event.astype(int))
    data = stats.CensoredData.right_censored(duration, ~event)
    expected = stats.ecdf(data).sf.evaluate(km["time"].to_numpy())
    np.testing.assert_allclose(km["survival"].to_numpy(), expected)


def test_kaplan_meier_refuses_an_exit_before_entry():
    with pytest.raises(ValueError, match="entry"):
        kaplan_meier([1, 2], [1, 0], entry=[0, 2])


def test_cumulative_incidence_by_hand_and_it_adds_up():
    cif = cumulative_incidence([1, 2, 3, 4], ["failed", "left", "failed", "reporting"]).set_index(
        "time"
    )
    assert cif.loc[4, "cif_failed"] == pytest.approx(0.5)
    assert cif.loc[4, "cif_left"] == pytest.approx(0.25)
    assert (cif["survival"] + cif["cif_failed"] + cif["cif_left"]).to_numpy() == pytest.approx(1.0)


def test_cumulative_incidence_with_one_cause_is_one_minus_kaplan_meier():
    rng = np.random.default_rng(7)
    duration = rng.integers(1, 15, size=300)
    cause = np.where(rng.random(300) < 0.3, "failed", "reporting")
    cif = cumulative_incidence(duration, cause, causes=("failed",))
    km = kaplan_meier(duration, (cause == "failed").astype(int))
    np.testing.assert_allclose(cif["cif_failed"], 1 - km["survival"])
