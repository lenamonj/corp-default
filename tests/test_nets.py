import random

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import roc_auc_score

from corp_default.nets import (
    SequenceHazardNet,
    TabularNet,
    build_sequences,
    cumulative_pd,
    hazard_loss,
    predict,
    seed_everything,
    train_net,
)


def company(name: str, years: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "company_name": name,
            "year": years,
            "a": [float(year) for year in years],
            "b": [float(year) * 10 for year in years],
        }
    )


def planted(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Five inputs, the first two carry the signal; (x, target, observed)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5)).astype(np.float32)
    y = (x[:, 0] + x[:, 1] + 0.3 * rng.normal(size=n) > 1.0).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y), torch.ones(n)


def fit_tabular(seed: int) -> np.ndarray:
    seed_everything(seed)
    model = TabularNet(n_inputs=5)
    train_net(
        model,
        planted(2000, 0),
        planted(500, 1),
        epochs=30,
        lr=0.01,
        weight_decay=1e-4,
        patience=5,
        seed=seed,
    )
    return predict(model, planted(500, 2)[0])


def test_seed_everything_repeats_python_numpy_and_torch():
    seed_everything(7)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    seed_everything(7)
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == first
    assert torch.are_deterministic_algorithms_enabled()


def test_a_reporting_gap_gives_masked_steps():
    frame = company("C_1", [2001, 2002, 2005])
    values, mask = build_sequences(frame, ["a", "b"], window=5)
    assert values.shape == (3, 5, 2) and mask.shape == (3, 5)
    # the 2005 row's window is 2001, 2002, 2003, 2004, 2005, oldest first
    assert mask[2].tolist() == [True, True, False, False, True]
    assert values[2, :, 0].tolist() == [2001.0, 2002.0, 0.0, 0.0, 2005.0]
    assert values[2, :, 1].tolist() == [20010.0, 20020.0, 0.0, 0.0, 20050.0]


def test_a_company_with_one_year_gives_one_unmasked_step():
    values, mask = build_sequences(company("C_1", [2003]), ["a", "b"], window=5)
    assert mask.tolist() == [[False, False, False, False, True]]
    assert values[0, -1].tolist() == [2003.0, 20030.0]


def test_sequences_never_cross_companies_and_keep_the_row_order():
    frame = pd.concat([company("C_2", [2002, 2001]), company("C_1", [2001])], ignore_index=True)
    values, mask = build_sequences(frame, ["a"], window=2)
    assert mask.tolist() == [[True, True], [False, True], [False, True]]
    assert values[:, :, 0].tolist() == [[2001.0, 2002.0], [0.0, 2001.0], [0.0, 2001.0]]


def test_build_sequences_refuses_a_repeated_company_year():
    with pytest.raises(ValueError, match="one row per"):
        build_sequences(company("C_1", [2001, 2001]), ["a"])


def test_cumulative_pd_by_hand_and_never_decreases():
    assert cumulative_pd(np.array([[0.1, 0.2, 0.5]])) == pytest.approx(
        np.array([[0.1, 0.28, 0.64]])
    )
    hazards = np.random.default_rng(0).uniform(size=(200, 3))
    assert (np.diff(cumulative_pd(hazards), axis=1) >= 0).all()


def test_hazard_loss_by_hand():
    # logits of zero give a loss of log 2 on every observed cell
    loss = hazard_loss(torch.zeros(2, 3), torch.ones(2, 3), torch.ones(2, 3))
    assert loss.item() == pytest.approx(np.log(2))


def test_hazard_loss_ignores_an_unobserved_target():
    logits = torch.tensor([[0.5, -1.0, 2.0], [0.1, 0.2, 0.3]])
    targets = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    observed = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    flipped = targets.clone()
    flipped[0, 2] = 1.0
    assert hazard_loss(logits, targets, observed) == hazard_loss(logits, flipped, observed)
    seen = targets.clone()
    seen[0, 1] = 0.0
    assert hazard_loss(logits, targets, observed) != hazard_loss(logits, seen, observed)


def test_sequence_net_gives_three_logits_and_ignores_masked_steps():
    seed_everything(0)
    model = SequenceHazardNet(n_inputs=4).eval()
    x = torch.randn(6, 5, 4)
    mask = torch.tensor([[True, True, False, False, True]] * 6)
    changed = x.clone()
    changed[:, 2:4] = 99.0
    with torch.no_grad():
        assert model(x, mask).shape == (6, 3)
        assert torch.equal(model(x, mask), model(changed, mask))
        assert not torch.equal(model(x, mask), model(x, torch.ones(6, 5, dtype=torch.bool)))


def test_two_trainings_with_the_same_seed_give_identical_predictions():
    assert np.array_equal(fit_tabular(3), fit_tabular(3))
    assert not np.array_equal(fit_tabular(3), fit_tabular(4))


def test_tabular_net_learns_a_planted_signal():
    scores = fit_tabular(42)
    assert roc_auc_score(planted(500, 2)[1].numpy(), scores) > 0.9


def test_train_net_stops_early_and_returns_the_best_state():
    seed_everything(0)
    model = TabularNet(n_inputs=5)
    x, y, observed = planted(400, 0)
    noise = (x, torch.from_numpy(np.random.default_rng(5).permutation(y.numpy())), observed)
    result = train_net(
        model, noise, planted(400, 1), epochs=200, lr=0.01, weight_decay=0.0, patience=3, seed=0
    )
    history = result["history"]
    assert len(history) < 200
    assert len(history) == result["best_epoch"] + 3
    assert history["valid_loss"].idxmin() + 1 == result["best_epoch"]
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, result["state"][name])


def test_train_net_without_a_stopping_set_runs_every_epoch_and_keeps_the_last_state():
    seed_everything(0)
    model = TabularNet(n_inputs=5)
    result = train_net(
        model, planted(400, 0), None, epochs=4, lr=0.01, weight_decay=0.0, patience=1, seed=0
    )
    assert list(result["history"]["epoch"]) == [1, 2, 3, 4]
    assert result["history"]["valid_loss"].isna().all()
    assert result["best_epoch"] == 4
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, result["state"][name])


def test_sequence_net_learns_a_planted_trend():
    def trend(n: int, seed: int) -> tuple[torch.Tensor, ...]:
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, 5, 2)).astype(np.float32)
        y = (x[:, -1, 0] - x[:, 0, 0] > 1.0).astype(np.float32)
        return (
            torch.from_numpy(x),
            torch.ones(n, 5, dtype=torch.bool),
            torch.from_numpy(np.repeat(y[:, None], 3, axis=1)),
            torch.ones(n, 3),
        )

    seed_everything(0)
    model = SequenceHazardNet(n_inputs=2)
    train_net(
        model,
        trend(3000, 0),
        trend(500, 1),
        epochs=40,
        lr=0.01,
        weight_decay=0.0,
        patience=5,
        seed=0,
    )
    x, mask, y, _ = trend(500, 2)
    assert roc_auc_score(y[:, 0].numpy(), predict(model, x, mask)[:, 0]) > 0.9
