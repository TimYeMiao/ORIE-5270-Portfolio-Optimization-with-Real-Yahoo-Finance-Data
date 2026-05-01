import yfinance as yf
import pandas as pd


def download_price_data(tickers, start_date, end_date):
    """
    Download adjusted close price data from Yahoo Finance.

    Parameters
    ----------
    tickers : list of str
        List of stock tickers.
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        Adjusted close prices with dates as index and tickers as columns.
    """
    if not tickers:
        raise ValueError("tickers must be a non-empty list")

    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = tickers

    prices = prices.dropna(how="all")
    prices = prices.ffill().dropna()

    return prices