from enum import Enum, unique
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import cross_val_predict

from omics_tools import multivariate


@unique
class LvmMethod(Enum):
    NMF = "NMF"
    ICA = "ICA"
    PCA = "PCA"


def lvm_reproducibility(
    data: pd.DataFrame,
    n_components: int,
    n_random_states: int,
    lvm_method: LvmMethod,
    **kwargs,
) -> pd.DataFrame:
    """Evaluates the reproducibility of latent variable model (LVM) runs.

    Runs an LVM model multiple times with different random states, keeping all
    other parameters constant. It then calculates the maximum correlation
    between each of the components of given run and all other components of
    another run.

    Args:
        data: The input data to perform ICA on.
        n_components: The number of components to decompose the data into.
        n_random_states: The number of different random states.
        lvm_method: The LVM method. Either `LvmMethod.ICA` or `LvmMethod.NMF` or
            `LvmMethod.PCA`.
        **kwargs: Additional keyword arguments to pass to the model function.

    Returns:
        A DataFrame containing the following columns:
        - n_components: the number of components.
        - component_a: the component of run `a`.
        - random_state_a: the random state of run `a`.
        - random_state_b: the random state of of run `b`.
        - r2_max: the maximum correlation of `component_a` across all components of run `b`.

    Raises:
        ValueError: If an invalid `lvm_method` is provided.

    Example:
        >>> import pandas as pd
        >>> from sklearn.datasets import load_iris
        >>> from omics_tools.evaluation import lvm_reproducibility
        >>> data = pd.DataFrame(load_iris().data)
        >>> result = lvm_reproducibility(data, n_components=3, n_random_states=5,
        >>>                              lvm_method=LvmMethod.ICA)
        >>> print(result.head())
    """

    if lvm_method == LvmMethod.NMF:
        lvm_f = multivariate.nmf
    elif lvm_method == LvmMethod.ICA:
        lvm_f = multivariate.ica
    elif lvm_method == LvmMethod.PCA:
        lvm_f = multivariate.pca
    else:
        raise ValueError("Invalid method")

    results = pd.concat(
        [
            lvm_f(
                data=data,
                n_components=n_components,
                random_state=random_state,
                **kwargs,
            ).feature_weights
            for random_state in range(n_random_states)
        ]
    )

    corr_components = pd.DataFrame(
        results.T.corr(method="spearman"), columns=results.index, index=results.index
    ).abs()
    corr_components = (
        corr_components.reset_index()
        .melt(id_vars="index", var_name="component_b", value_name="r2")
        .rename(columns={"index": "component_a"})
    )
    corr_components[["n_components", "random_state_a", "component_a"]] = (
        corr_components["component_a"].str.split("_", expand=True).astype(int)
    )
    corr_components[["n_components", "random_state_b", "component_b"]] = (
        corr_components["component_b"].str.split("_", expand=True).astype(int)
    )
    corr_components = corr_components[
        corr_components["random_state_a"] != corr_components["random_state_b"]
    ]

    return (
        corr_components.groupby(
            ["n_components", "component_a", "random_state_a", "random_state_b"]
        )["r2"]
        .max()
        .to_frame()
        .reset_index()
        .rename(columns={"r2": "r2_max"})
    )


def cv_regression(
    model: Any,
    independent_variables: pd.DataFrame,
    dependent_variable: pd.Series,
    **kwargs: Any,
) -> pd.DataFrame:
    """Performs k-fold cross-validation (CV) on a regression model.

    Performns CV on a regression model and derives relevant model
    performance metrics (coefficient of determination, mean squared arror,
    mean absolute error).

    Args:
        model: The regression model to be evaluated.
        independent_variables: The independent variables.
        dependent_variable: The dependent variable (continuous).
        **kwargs: Additional parameters to pass to the `cross_val_predict` function from sklearn.

    Returns:
        A DataFrame with the following columns:
        - 'r2_scored': The coefficient of determination.
        - 'mse': The mean squared error.
        - 'mae': The mean absolute error.

    Example:
    >>> import pandas as pd
    >>> from sklearn.cross_decomposition import PLSRegression
    >>> from sklearn.datasets import load_diabetes
    >>> from omics_tools.multivariate import cv_regression
    >>> diabetes = load_diabetes()
    >>> independent_variables = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    >>> dependent_variable = pd.Series(diabetes.target)
    >>> result = cv_regression(PLSRegression(n_components=3), independent_variables, dependent_variable)
    >>> print(result)
    """

    # Process data
    not_null_mask = ~dependent_variable.isnull()
    dependent_variable = dependent_variable[not_null_mask]
    independent_variables = independent_variables[not_null_mask]
    independent_variables = independent_variables.fillna(independent_variables.mean())

    # Generate cross-validated predictions for each input data point
    dependent_variable_pred = cross_val_predict(
        estimator=model, X=independent_variables, y=dependent_variable, **kwargs
    )

    # Calculate metrics
    return pd.DataFrame(
        {
            "r2_score": [
                r2_score(y_true=dependent_variable, y_pred=dependent_variable_pred)
            ],
            "mse": [
                mean_squared_error(
                    y_true=dependent_variable, y_pred=dependent_variable_pred
                )
            ],
            "mae": [
                mean_absolute_error(
                    y_true=dependent_variable, y_pred=dependent_variable_pred
                )
            ],
        }
    )


def cv_classification(
    model: Any,
    independent_variables: pd.DataFrame,
    dependent_variable: pd.Series,
    round_predictions: bool = False,
    average: str = "binary",
    **kwargs: Any,
) -> pd.DataFrame:
    """Performs k-fold cross-validation (CV) on a classification model.

    Performs CV on a classification model and derives relevant model
    performance metrics (accuracy, precision, recall, F1-score).

    Args:
        model: The classification model to be evaluated.
        independent_variables: The independent variables.
        dependent_variable: The dependent variable.
        round_predictions: Whether to round up predictions to nearest integer.
        average: The averaging strategy for metrics when the dependent variable has
            more than 2 different labels. Options: 'micro', 'macro', 'samples', 'weighted',
            'binary' or None. For details see the "precision_score" function from sklearn.
        **kwargs: Additional parameters to pass to the `cross_val_predict` function from sklearn.

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
          - 'accuracy': The accuracy of the model.
          - 'precision': The precision of the model.
          - 'recall': The recall of the model.
          - 'f1_score': The F1-score of the model.

    Example:
    >>> import pandas as pd
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import load_breast_cancer
    >>> from omics_tools.multivariate import cv_classification
    >>> cancer = load_breast_cancer()
    >>> independent_variables = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    >>> dependent_variable = pd.Series(cancer.target)
    >>> result = cv_classification(LogisticRegression(max_iter=1000), independent_variables, dependent_variable)
    >>> print(result)
    """

    # Process data
    not_null_mask = ~dependent_variable.isnull()
    dependent_variable = dependent_variable[not_null_mask]
    independent_variables = independent_variables[not_null_mask]
    independent_variables = independent_variables.fillna(independent_variables.mean())

    # Generate cross-validated predictions for each input data point
    dependent_variable_pred = cross_val_predict(
        estimator=model, X=independent_variables, y=dependent_variable, **kwargs
    )

    # Optionally round up predictions
    if round_predictions:
        dependent_variable_pred = dependent_variable_pred.round()

    return pd.DataFrame(
        {
            "accuracy": [
                accuracy_score(
                    y_true=dependent_variable, y_pred=dependent_variable_pred
                )
            ],
            "precision": [
                precision_score(
                    y_true=dependent_variable,
                    y_pred=dependent_variable_pred,
                    average=average,
                )
            ],
            "recall": [
                recall_score(
                    y_true=dependent_variable,
                    y_pred=dependent_variable_pred,
                    average=average,
                )
            ],
            "f1_score": [
                f1_score(
                    y_true=dependent_variable,
                    y_pred=dependent_variable_pred,
                    average=average,
                )
            ],
        }
    )
