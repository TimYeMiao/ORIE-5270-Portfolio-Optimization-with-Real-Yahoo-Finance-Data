import numpy as np
import pytest

from portfolio_project.optimization import (
    equal_weight_portfolio,
    minimum_variance_portfolio,
    maximum_sharpe_portfolio,
)


def test_equal_weight_portfolio_sums_to_one():
    weights = equal_weight_portfolio(4)

    assert len(weights) == 4
    assert abs(weights.sum() - 1.0) < 1e-8


def test_equal_weight_portfolio_invalid_n_assets():
    with pytest.raises(ValueError):
        equal_weight_portfolio(0)


def test_minimum_variance_weights_sum_to_one():
    cov_matrix = np.array([
        [0.04, 0.01],
        [0.01, 0.09],
    ])

    weights = minimum_variance_portfolio(cov_matrix)

    assert abs(weights.sum() - 1.0) < 1e-6
    assert np.all(weights >= -1e-8)


def test_maximum_sharpe_weights_sum_to_one():
    mean_returns = np.array([0.001, 0.0005])
    cov_matrix = np.array([
        [0.04, 0.01],
        [0.01, 0.09],
    ])

    weights = maximum_sharpe_portfolio(mean_returns, cov_matrix)

    assert abs(weights.sum() - 1.0) < 1e-6
    assert np.all(weights >= -1e-8)