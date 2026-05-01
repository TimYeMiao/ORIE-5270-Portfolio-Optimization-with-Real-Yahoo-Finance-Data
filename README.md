# Reproducible Portfolio Optimization with Yahoo Finance Data

## Project Overview

This project builds a reproducible Python package for portfolio optimization using real historical stock price data from Yahoo Finance. The goal is to compare different portfolio construction methods and evaluate their out-of-sample performance.

The project focuses on three portfolio strategies:

1. Equal-weight portfolio
2. Minimum-variance portfolio
3. Maximum-Sharpe portfolio

The workflow includes data downloading, return calculation, train-test splitting, portfolio optimization, backtesting, performance evaluation, unit testing, and notebook-based demonstration.

---

## Project Purpose

The purpose of this project is to provide a clean and reproducible implementation of portfolio optimization methods using real financial data.

Specifically, this project answers the following questions:

- How can we construct a simple benchmark portfolio using equal weights?
- How can we build a minimum-variance portfolio based on the estimated covariance matrix?
- How can we build a maximum-Sharpe portfolio using estimated returns and covariance?
- How do these portfolios perform out of sample?
- How can we organize a financial computation project as a reusable Python package with unit tests?

This project is designed for ORIE 5270 and emphasizes reproducibility, code quality, documentation, and unit-test coverage.

---

## Dataset

The project uses daily stock price data from Yahoo Finance through the `yfinance` Python package.

The selected stocks are large-cap U.S. companies from different sectors:

- AAPL
- MSFT
- AMZN
- GOOGL
- JPM
- XOM
- JNJ
- PG

The sample period is:

```text
2018-01-01 to 2024-12-31
```

The train-test split date is:

```text
2022-01-01
```

Data before 2022-01-01 is used as the training period. Data from 2022 onward is used for out-of-sample backtesting.

The package downloads adjusted price data using `auto_adjust=True`, so the price series accounts for corporate actions such as dividends and stock splits.

---

## Methodology

The project follows this pipeline:

```text
Download price data
→ Compute daily returns
→ Split data into training and testing periods
→ Estimate mean returns and covariance matrix using the training data
→ Construct portfolios
→ Backtest portfolios on the testing data
→ Evaluate performance metrics
```

The three portfolio strategies are described below.

### 1. Equal-Weight Portfolio

The equal-weight portfolio assigns the same weight to each asset. It serves as a simple benchmark because it does not rely on estimated returns or covariance optimization.

### 2. Minimum-Variance Portfolio

The minimum-variance portfolio solves an optimization problem that minimizes portfolio variance subject to the following constraints:

```text
All weights are nonnegative
All weights sum to one
```

This creates a long-only portfolio that focuses on reducing risk.

### 3. Maximum-Sharpe Portfolio

The maximum-Sharpe portfolio maximizes the Sharpe ratio based on estimated mean returns and covariance matrix from the training period.

This method considers both expected return and risk.

---

## Performance Metrics

The out-of-sample performance is evaluated using:

- Annualized return
- Annualized volatility
- Sharpe ratio
- Maximum drawdown
- Cumulative return

These metrics are implemented in the `metrics.py` module.

---

## File Structure

```text
portfolio_project/
│
├── README.md
├── requirements.txt
├── pyproject.toml
├── conda.yaml
│
├── src/
│   └── portfolio_project/
│       ├── __init__.py
│       ├── data.py
│       ├── features.py
│       ├── optimization.py
│       ├── backtest.py
│       └── metrics.py
│
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_optimization.py
│   ├── test_backtest.py
│   └── test_metrics.py
│
├── scripts/
│   └── run_pipeline.py
│
└── notebooks/
    └── demo.ipynb
```

---

## Installation

First, clone this repository:

```bash
git clone https://github.com/TimYeMiao/ORIE-5270-Portfolio-Optimization-with-Real-Yahoo-Finance-Data.git
cd ORIE-5270-Portfolio-Optimization-with-Real-Yahoo-Finance-Data
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

Install the project as an editable Python package:

```bash
python -m pip install -e .
```

Alternatively, users can create a conda environment:

```bash
conda env create -f conda.yaml
conda activate portfolio-project
python -m pip install -e .
```

After installation, modules can be imported as:

```python
from portfolio_project.features import compute_returns
from portfolio_project.optimization import minimum_variance_portfolio
```

---

## Running the Project

To run the full pipeline from the command line:

```bash
python scripts/run_pipeline.py
```

This script will:

1. Download stock price data from Yahoo Finance
2. Compute daily returns
3. Split the data into training and testing periods
4. Construct equal-weight, minimum-variance, and maximum-Sharpe portfolios
5. Backtest the portfolios
6. Print a performance summary
7. Plot out-of-sample cumulative returns

---

## Running the Notebook Demo

A notebook demonstration is provided in:

```text
notebooks/demo.ipynb
```

The notebook shows the full workflow step by step, including:

- Data downloading
- Price visualization
- Return calculation
- Portfolio weight comparison
- Out-of-sample cumulative return plot
- Performance summary table
- Interpretation of results

To run the notebook, first install the package using:

```bash
python -m pip install -e .
```

Then open `notebooks/demo.ipynb` and run all cells.

---

## Running Unit Tests

Unit tests are included in the `tests/` folder. To run all tests:

```bash
python -m pytest
```

To check test coverage:

```bash
python -m pytest --cov=portfolio_project
```

The current implementation has total test coverage above 80%. The tests cover the core computational components, including:

- Return calculation
- Train-test splitting
- Portfolio optimization
- Backtesting
- Performance metrics
- Basic data input validation

The data downloading function depends on the external Yahoo Finance API, so it is tested mainly for input validation rather than live API behavior.

---

## Example Results

The out-of-sample cumulative return plot compares the three strategies during the testing period.

In the current implementation, the equal-weight portfolio, minimum-variance portfolio, and maximum-Sharpe portfolio show different risk-return profiles. The equal-weight portfolio provides a simple benchmark. The minimum-variance portfolio focuses on reducing risk, while the maximum-Sharpe portfolio attempts to improve risk-adjusted performance by using both estimated returns and covariance.

The final results may vary slightly depending on Yahoo Finance data updates and package versions.

In the current run, the equal-weight portfolio achieves the highest cumulative return during the testing period, while the maximum-Sharpe portfolio also performs strongly after the market recovery period. The minimum-variance portfolio is more conservative and shows lower cumulative return, which is consistent with its objective of reducing portfolio variance rather than maximizing return.

---

## Interpretation

The equal-weight portfolio is simple and does not rely on parameter estimation. This makes it a useful benchmark.

The minimum-variance portfolio uses the covariance matrix estimated from the training data. It is designed to reduce portfolio volatility, but it may not always generate the highest return.

The maximum-Sharpe portfolio uses both expected returns and covariance estimates. It can produce stronger risk-adjusted performance, but it may also be more sensitive to estimation error.

Overall, the project demonstrates how portfolio optimization methods can be implemented, tested, and evaluated in a reproducible Python package.

---

## Limitations

This project has several limitations:

1. The expected returns and covariance matrix are estimated from historical data and may be unstable.
2. The backtest does not include transaction costs, taxes, or liquidity constraints.
3. The portfolios are constructed using a fixed train-test split rather than a rolling-window rebalancing framework.
4. The strategy uses long-only constraints and does not consider short selling.

---

## Future Improvements

Possible extensions include:

- Rolling-window portfolio rebalancing
- Transaction cost modeling
- Alternative covariance estimation methods
- Risk-parity portfolio construction
- Sector constraints
- More asset classes, such as ETFs, bonds, or commodities
- Comparison with market benchmarks such as SPY

---

## Author

Ye Miao