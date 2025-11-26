"""Reusable utilities for the Customer Satisfaction project.

Provides functions for loading data, summarizing variables, running EFA (factor analysis),
computing KMO/Bartlett, generating factor scores, and fitting simple linear regressions
on factor scores. These helpers mirror common steps in `costomer_satisfaction.ipynb`.

Dependencies: pandas, numpy, sklearn, factor_analyzer, scipy, statsmodels (optional)
"""

from typing import Tuple, Dict, Optional
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

try:
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
except Exception:
    FactorAnalyzer = None
    calculate_kmo = None
    calculate_bartlett_sphericity = None


def load_data(csv_path: str, parse_dates: Optional[list] = None) -> pd.DataFrame:
    """Load CSV into a pandas DataFrame.

    Args:
        csv_path: path to CSV file.
        parse_dates: list of columns to parse as dates.

    Returns:
        pd.DataFrame
    """
    return pd.read_csv(csv_path, parse_dates=parse_dates)


def data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary table with mean, std, min, max and percent missing for numeric cols.

    Args:
        df: input DataFrame

    Returns:
        DataFrame with summary statistics indexed by column name.
    """
    num = df.select_dtypes(include=[np.number])
    summary = pd.DataFrame({
        "count": num.count(),
        "mean": num.mean(),
        "std": num.std(),
        "min": num.min(),
        "max": num.max(),
        "%missing": num.isna().mean() * 100
    })
    return summary


def compute_kmo(df: pd.DataFrame) -> Tuple[Optional[float], Optional[pd.Series]]:
    """Compute the Kaiser-Meyer-Olkin (KMO) measure for df.

    Returns overall_kmo, kmo_per_item. If factor_analyzer is not installed returns (None, None).
    """
    if calculate_kmo is None:
        return None, None
    numeric = df.select_dtypes(include=[np.number]).dropna()
    kmo_all, kmo_per_item = calculate_kmo(numeric)
    return float(kmo_all), pd.Series(kmo_per_item, index=numeric.columns)


def compute_bartlett(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """Compute Bartlett's test of sphericity. Returns (chi_square, p_value).

    If factor_analyzer not installed returns (None, None).
    """
    if calculate_bartlett_sphericity is None:
        return None, None
    numeric = df.select_dtypes(include=[np.number]).dropna()
    chi_square_value, p_value = calculate_bartlett_sphericity(numeric)
    return float(chi_square_value), float(p_value)


def run_efa(df: pd.DataFrame, n_factors: int = 5, rotation: Optional[str] = "varimax", method: str = "principal") -> Dict:
    """Run exploratory factor analysis and return results.

    Returns a dict with keys: 'fa' (FactorAnalyzer object), 'loadings' (DataFrame),
    'variance' (variance, prop_var, cum_var), 'scores' (DataFrame of factor scores).

    Note: Input df should already be preprocessed (numeric, standardized or not depending on preference).
    """
    if FactorAnalyzer is None:
        raise ImportError("factor_analyzer is required for EFA. Install via pip install factor_analyzer")

    numeric = df.select_dtypes(include=[np.number]).dropna()
    fa = FactorAnalyzer(n_factors=n_factors, method=method, rotation=rotation)
    fa.fit(numeric)

    loadings = pd.DataFrame(fa.loadings_, index=numeric.columns, columns=[f"Factor{i+1}" for i in range(n_factors)])
    variance, prop_var, cum_var = fa.get_factor_variance()

    # factor scores (if supported)
    try:
        scores = pd.DataFrame(fa.transform(numeric), index=numeric.index, columns=[f"Factor{i+1}" for i in range(n_factors)])
    except Exception:
        scores = pd.DataFrame()

    return {
        "fa": fa,
        "loadings": loadings,
        "variance": {
            "eigenvalues": variance,
            "prop_var": prop_var,
            "cum_var": cum_var
        },
        "scores": scores
    }


def regress_on_factors(X: pd.DataFrame, y: pd.Series) -> Dict:
    """Fit a simple linear regression of y on X (factor scores) and return metrics.

    Returns dict with keys: model, coefficients (pd.Series), intercept, r2, rmse, y_pred
    """
    X_num = X.select_dtypes(include=[np.number])
    model = LinearRegression()
    model.fit(X_num, y)
    y_pred = model.predict(X_num)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    coefs = pd.Series(model.coef_, index=X_num.columns)

    return {
        "model": model,
        "coefficients": coefs,
        "intercept": float(model.intercept_),
        "r2": float(r2),
        "rmse": float(rmse),
        "y_pred": pd.Series(y_pred, index=X_num.index)
    }


def export_dataframe(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV (UTF-8, index preserved)."""
    df.to_csv(path, index=True)


if __name__ == "__main__":
    # simple smoke test when run directly (won't execute EFA if factor_analyzer missing)
    print("utils.py: helper module for customer satisfaction project")
