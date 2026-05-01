import numpy as np
import pandas as pd


def annualized_return(returns, periods_per_year=252):
    """
    Compute annualized return.
    """
    cumulative_growth = (1 + returns).prod()
    n_periods = len(returns)

    if n_periods == 0:
        raise ValueError("returns cannot be empty")

    return cumulative_growth ** (periods_per_year / n_periods) - 1


def annualized_volatility(returns, periods_per_year=252):
    """
    Compute annualized volatility.
    """
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Compute annualized Sharpe ratio.
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    volatility = annualized_volatility(excess_returns, periods_per_year)

    if volatility == 0:
        return np.nan

    return annualized_return(excess_returns, periods_per_year) / volatility


def max_drawdown(returns):
    """
    Compute maximum drawdown.
    """
    cumulative_value = (1 + returns).cumprod()
    running_max = cumulative_value.cummax()
    drawdown = cumulative_value / running_max - 1

    return drawdown.min()


def performance_summary(returns_dict):
    """
    Create a performance summary table for multiple strategies.

    Parameters
    ----------
    returns_dict : dict
        Dictionary mapping strategy names to return series.

    Returns
    -------
    pd.DataFrame
        Summary table.
    """
    rows = []

    for name, returns in returns_dict.items():
        rows.append({
            "Strategy": name,
            "Annual Return": annualized_return(returns),
            "Annual Volatility": annualized_volatility(returns),
            "Sharpe Ratio": sharpe_ratio(returns),
            "Max Drawdown": max_drawdown(returns),
        })

    return pd.DataFrame(rows).set_index("Strategy")