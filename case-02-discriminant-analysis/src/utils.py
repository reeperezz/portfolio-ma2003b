
"""
utils.py - Helper functions for the LendSmart Credit Risk Analysis project.

This module centralizes common tasks:
- Data loading
- Encoding and preprocessing
- Train/test split with scaling
- Model training (LDA, QDA)
- Evaluation metrics and ROC curves
"""

from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

# Data loading & preprocessing

def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load the LendSmart credit risk dataset from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file (e.g. 'data/raw/credit_risk_data-1.csv').

    Returns
    -------
    df : pd.DataFrame
        Loaded dataframe.
    """
    df = pd.read_csv(csv_path)
    return df


def encode_categorical(
    df: pd.DataFrame,
    categorical_cols: List[str] = None,
    drop_first: bool = True,
) -> pd.DataFrame:
    """
    One-hot encode categorical columns using pandas.get_dummies.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe.
    categorical_cols : list of str, optional
        Columns to encode. If None, only 'education_level' and
        'marital_status' will be encoded (the ones used in the project).
    drop_first : bool, default True
        Whether to drop the first category to avoid dummy-variable trap.

    Returns
    -------
    df_encoded : pd.DataFrame
        Dataframe with dummy variables.
    """
    if categorical_cols is None:
        categorical_cols = ["education_level", "marital_status"]

    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first)
    return df_encoded


def build_feature_target_matrices(
    df_encoded: pd.DataFrame,
    target_col: str = "loan_status",
    drop_cols: List[str] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate predictors X and target y from the encoded dataframe.

    Parameters
    ----------
    df_encoded : pd.DataFrame
        Encoded dataframe.
    target_col : str, default "loan_status"
        Column name of the target.
    drop_cols : list of str, optional
        Columns to drop from X (IDs, dates, etc.).
        By default, drops 'application_id' and 'application_date'.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    """
    if drop_cols is None:
        drop_cols = ["application_id", "application_date"]

    X = df_encoded.drop(drop_cols + [target_col], axis=1)
    y = df_encoded[target_col]
    return X, y


def train_test_split_scaled(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """
    Perform stratified train/test split and scale features with StandardScaler.

    The scaler is fitted only on the training set to avoid data leakage.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    test_size : float, default 0.2
        Proportion of data for testing.
    random_state : int, default 42
        Random seed.

    Returns
    -------
    X_train_scaled : pd.DataFrame
    X_test_scaled : pd.DataFrame
    y_train : pd.Series
    y_test : pd.Series
    scaler : StandardScaler
        Fitted scaler (useful if you need to transform new data later).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Keep DataFrame structure (column names & indexes)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ---------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------


def fit_lda(X_train: pd.DataFrame, y_train: pd.Series) -> LinearDiscriminantAnalysis:
    """
    Train a Linear Discriminant Analysis (LDA) model.

    Returns
    -------
    lda : LinearDiscriminantAnalysis
        Fitted LDA model.
    """
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    return lda


def fit_qda(X_train: pd.DataFrame, y_train: pd.Series) -> QuadraticDiscriminantAnalysis:
    """
    Train a Quadratic Discriminant Analysis (QDA) model.

    Returns
    -------
    qda : QuadraticDiscriminantAnalysis
        Fitted QDA model.
    """
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    return qda


# ---------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Evaluate a classifier on a test set.

    Parameters
    ----------
    model : sklearn-like estimator
        Must implement predict() (and ideally predict_proba()).
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True labels.
    verbose : bool, default True
        If True, prints report and confusion matrix.

    Returns
    -------
    results : dict
        Dictionary with accuracy, classification_report (str),
        confusion_matrix (np.ndarray) and predictions.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    if verbose:
        print("Classification report")
        print(report)
        print(f"Accuracy: {acc:.3f}")

        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

    return {
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "y_pred": y_pred,
    }


def cross_validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
) -> Tuple[float, float]:
    """
    Perform cross-validation and return mean and std of accuracy.

    Parameters
    ----------
    model : sklearn-like estimator
        Estimator to cross-validate (typically wrapped in a Pipeline).
    X : pd.DataFrame
        Features.
    y : pd.Series
        Target.
    cv : int, default 5
        Number of folds.

    Returns
    -------
    mean_acc : float
    std_acc : float
    """
    scores = cross_val_score(model, X, y, cv=cv)
    mean_acc = scores.mean()
    std_acc = scores.std()
    print(f"CV Accuracy: {mean_acc:.3f} (+/- {std_acc:.3f})")
    return mean_acc, std_acc


def plot_roc_curves(
    models: Dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figsize: Tuple[int, int] = (8, 6),
    save_path: str = None,
) -> None:
    """
    Plot ROC curves for multiple models on the same figure.

    Parameters
    ----------
    models : dict
        Keys are labels (e.g. 'LDA', 'QDA'), values are fitted classifiers
        implementing predict_proba().
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True labels.
    figsize : tuple, default (8, 6)
        Size of the matplotlib figure.
    save_path : str, optional
        If provided, save the figure to this path.
    """
    plt.figure(figsize=figsize)

    for name, clf in models.items():
        y_prob = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()


def lda_feature_importance(
    lda_model: LinearDiscriminantAnalysis,
    feature_names: List[str],
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Get a sorted DataFrame of LDA coefficients as a proxy for feature importance.

    Parameters
    ----------
    lda_model : LinearDiscriminantAnalysis
        Fitted LDA model.
    feature_names : list of str
        Names of features in the same order as used for training.
    top_n : int, default 10
        Number of top features to return.

    Returns
    -------
    coef_df : pd.DataFrame
        DataFrame with columns ['Feature', 'Coefficient', 'Abs_Coefficient'].
    """
    coef = lda_model.coef_[0]
    coef_df = pd.DataFrame(
        {"Feature": feature_names, "Coefficient": coef}
    )
    coef_df["Abs_Coefficient"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("Abs_Coefficient", ascending=False)

    if top_n is not None:
        coef_df = coef_df.head(top_n)

    return coef_df
