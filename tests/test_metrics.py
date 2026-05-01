import numpy as np
import pandas as pd
import pytest

from portfolio_project.metrics import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    max_drawdown,
    performance_summary,
)


def test_annualized_return_positive():
    returns = pd.Series([0.01, 0.01, 0.01])

    result = annualized_return(returns)

    assert result > 0


def test_annualized_return_empty_returns():
    returns = pd.Series(dtype=float)

    with pytest.raises(ValueError):
        annualized_return(returns)


def test_annualized_volatility_positive():
    returns = pd.Series([0.01, -0.02, 0.03])

    result = annualized_volatility(returns)

    assert result > 0


def test_sharpe_ratio_valid():
    returns = pd.Series([0.01, 0.02, -0.01, 0.03])

    result = sharpe_ratio(returns)

    assert not np.isnan(result)


def test_sharpe_ratio_zero_volatility():
    returns = pd.Series([0.01, 0.01, 0.01])

    result = sharpe_ratio(returns)

    assert np.isnan(result)


def test_max_drawdown():
    returns = pd.Series([0.10, -0.20, 0.05])

    mdd = max_drawdown(returns)

    assert mdd < 0
    assert abs(mdd - (-0.20)) < 1e-8


def test_performance_summary():
    returns_dict = {
        "Strategy A": pd.Series([0.01, 0.02, -0.01]),
        "Strategy B": pd.Series([0.02, -0.01, 0.01]),
    }

    summary = performance_summary(returns_dict)

    assert "Annual Return" in summary.columns
    assert "Annual Volatility" in summary.columns
    assert "Sharpe Ratio" in summary.columns
    assert "Max Drawdown" in summary.columns
    assert summary.shape[0] == 2