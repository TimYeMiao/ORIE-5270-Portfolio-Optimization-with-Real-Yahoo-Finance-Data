import pandas as pd
import pytest

from portfolio_project.features import compute_returns, split_train_test


def test_compute_returns_shape():
    prices = pd.DataFrame({
        "AAPL": [100, 110, 121],
        "MSFT": [200, 220, 242],
    })

    returns = compute_returns(prices)

    assert returns.shape == (2, 2)


def test_compute_returns_values():
    prices = pd.DataFrame({
        "AAPL": [100, 110, 121],
    })

    returns = compute_returns(prices)

    assert abs(returns.iloc[0, 0] - 0.10) < 1e-8
    assert abs(returns.iloc[1, 0] - 0.10) < 1e-8


def test_compute_returns_empty_prices():
    prices = pd.DataFrame()

    with pytest.raises(ValueError):
        compute_returns(prices)


def test_split_train_test():
    returns = pd.DataFrame(
        {
            "AAPL": [0.01, 0.02, 0.03, 0.04],
            "MSFT": [0.02, 0.01, 0.04, 0.03],
        },
        index=pd.to_datetime([
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
            "2020-01-04",
        ])
    )

    train_returns, test_returns = split_train_test(returns, "2020-01-03")

    assert len(train_returns) == 2
    assert len(test_returns) == 2
    assert train_returns.index.max() < pd.Timestamp("2020-01-03")
    assert test_returns.index.min() >= pd.Timestamp("2020-01-03")


def test_split_train_test_empty_side():
    returns = pd.DataFrame(
        {
            "AAPL": [0.01, 0.02],
        },
        index=pd.to_datetime([
            "2020-01-01",
            "2020-01-02",
        ])
    )

    with pytest.raises(ValueError):
        split_train_test(returns, "2019-01-01")