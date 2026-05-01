import numpy as np
from scipy.optimize import minimize


def equal_weight_portfolio(n_assets):
    """
    Create equal-weight portfolio weights.

    Parameters
    ----------
    n_assets : int
        Number of assets.

    Returns
    -------
    np.ndarray
        Equal portfolio weights.
    """
    if n_assets <= 0:
        raise ValueError("n_assets must be positive")

    return np.ones(n_assets) / n_assets


def minimum_variance_portfolio(cov_matrix):
    """
    Compute minimum-variance portfolio weights.

    Parameters
    ----------
    cov_matrix : np.ndarray
        Covariance matrix of asset returns.

    Returns
    -------
    np.ndarray
        Optimized portfolio weights.
    """
    cov_matrix = np.asarray(cov_matrix)
    n_assets = cov_matrix.shape[0]

    def portfolio_variance(weights):
        # Scale the objective to improve numerical stability for daily returns
        return 10000 * (weights.T @ cov_matrix @ weights)

    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
    bounds = [(0, 1) for _ in range(n_assets)]
    initial_weights = equal_weight_portfolio(n_assets)

    result = minimize(
        portfolio_variance,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    if not result.success:
        raise RuntimeError("Minimum variance optimization failed")

    return result.x


def maximum_sharpe_portfolio(mean_returns, cov_matrix, risk_free_rate=0.0):
    """
    Compute maximum-Sharpe portfolio weights.

    Parameters
    ----------
    mean_returns : np.ndarray
        Mean daily returns of assets.
    cov_matrix : np.ndarray
        Covariance matrix of asset returns.
    risk_free_rate : float
        Daily risk-free rate.

    Returns
    -------
    np.ndarray
        Optimized portfolio weights.
    """
    mean_returns = np.asarray(mean_returns)
    cov_matrix = np.asarray(cov_matrix)
    n_assets = len(mean_returns)

    def negative_sharpe(weights):
        portfolio_return = weights @ mean_returns
        portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)

        if portfolio_volatility == 0:
            return 1e6

        sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -sharpe

    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
    bounds = [(0, 1) for _ in range(n_assets)]
    initial_weights = equal_weight_portfolio(n_assets)

    result = minimize(
        negative_sharpe,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    if not result.success:
        raise RuntimeError("Maximum Sharpe optimization failed")

    return result.x