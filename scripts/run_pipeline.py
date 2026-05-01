import matplotlib.pyplot as plt

from portfolio_project.data import download_price_data
from portfolio_project.features import compute_returns, split_train_test
from portfolio_project.optimization import (
    equal_weight_portfolio,
    minimum_variance_portfolio,
    maximum_sharpe_portfolio,
)
from portfolio_project.backtest import (
    compute_portfolio_returns,
    compute_cumulative_returns,
)
from portfolio_project.metrics import performance_summary


def main():
    tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "XOM", "JNJ", "PG"]
    start_date = "2018-01-01"
    end_date = "2024-12-31"
    split_date = "2022-01-01"

    prices = download_price_data(tickers, start_date, end_date)
    returns = compute_returns(prices)

    train_returns, test_returns = split_train_test(returns, split_date)

    mean_returns = train_returns.mean().values
    cov_matrix = train_returns.cov().values

    equal_weights = equal_weight_portfolio(len(tickers))
    min_var_weights = minimum_variance_portfolio(cov_matrix)
    max_sharpe_weights = maximum_sharpe_portfolio(mean_returns, cov_matrix)

    strategy_returns = {
        "Equal Weight": compute_portfolio_returns(test_returns, equal_weights),
        "Minimum Variance": compute_portfolio_returns(test_returns, min_var_weights),
        "Maximum Sharpe": compute_portfolio_returns(test_returns, max_sharpe_weights),
    }

    summary = performance_summary(strategy_returns)
    print(summary)

    for name, returns_series in strategy_returns.items():
        cumulative_returns = compute_cumulative_returns(returns_series)
        plt.plot(cumulative_returns, label=name)

    plt.title("Out-of-Sample Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()