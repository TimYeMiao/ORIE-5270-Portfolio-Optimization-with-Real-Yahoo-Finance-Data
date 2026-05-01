import numpy as np
import pandas as pd
import pytest

from portfolio_project.backtest import (
    compute_portfolio_returns,
    compute_cumulative_returns,
)


def test_compute_portfolio_returns():
    returns = pd.DataFrame({
        "AAPL": [0.01, 0.02],
        "MSFT": [0.03, 0.04],
    })
    weights = np.array([0.5, 0.5])

    portfolio_returns = compute_portfolio_returns(returns, weights)

    assert abs(portfolio_returns.iloc[0] - 0.02) < 1e-8
    assert abs(portfolio_returns.iloc[1] - 0.03) < 1e-8


def test_compute_portfolio_returns_wrong_weight_length():
    returns = pd.DataFrame({
        "AAPL": [0.01, 0.02],
        "MSFT": [0.03, 0.04],
    })
    weights = np.array([1.0])

    with pytest.raises(ValueError):
        compute_portfolio_returns(returns, weights)


def test_compute_cumulative_returns():
    returns = pd.Series([0.10, 0.10])

    cumulative_returns = compute_cumulative_returns(returns)

    assert abs(cumulative_returns.iloc[-1] - 0.21) < 1e-8