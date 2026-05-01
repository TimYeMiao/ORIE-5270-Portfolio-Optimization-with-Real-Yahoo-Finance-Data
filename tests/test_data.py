import pandas as pd
import pytest

from portfolio_project.data import download_price_data


def test_download_price_data_empty_tickers():
    with pytest.raises(ValueError):
        download_price_data([], "2020-01-01", "2020-01-10")