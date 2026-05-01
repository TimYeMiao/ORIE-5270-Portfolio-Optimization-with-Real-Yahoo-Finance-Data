import numpy as np
import pandas as pd


def compute_portfolio_returns(returns, weights):
    """
    Compute portfolio returns from asset returns and portfolio weights.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset return data.
    weights : np.ndarray
        Portfolio weights.

    Returns
    -------
    pd.Series
        Portfolio daily returns.
    """
    weights = np.asarray(weights)

    if returns.shape[1] != len(weights):
        raise ValueError("number of assets must match number of weights")

    portfolio_returns = returns @ weights

    return portfolio_returns


def compute_cumulative_returns(portfolio_returns):
    """
    Compute cumulative returns from daily portfolio returns.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio returns.

    Returns
    -------
    pd.Series
        Cumulative returns.
    """
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1

    return cumulative_returns