import pandas as pd

def compute_returns(prices):
    """
    Compute daily percentage returns from price data.

    Parameters
    ----------
    prices : pd.DataFrame
        Price data with dates as index and tickers as columns.

    Returns
    -------
    pd.DataFrame
        Daily returns.
    """
    if prices.empty:
        raise ValueError("prices cannot be empty")

    returns = prices.pct_change().dropna()

    return returns


def split_train_test(returns, split_date):
    """
    Split returns into training and testing sets.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily return data.
    split_date : str
        Date used to split train and test data.

    Returns
    -------
    tuple
        train_returns and test_returns.
    """
    train_returns = returns.loc[returns.index < split_date]
    test_returns = returns.loc[returns.index >= split_date]

    if train_returns.empty or test_returns.empty:
        raise ValueError("train or test set is empty")

    return train_returns, test_returns