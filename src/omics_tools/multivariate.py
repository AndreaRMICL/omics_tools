from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, FastICA
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
from sklearn.preprocessing import StandardScaler


class LvmResults(NamedTuple):
    """Aggregates the results from latent variable models (LVMs)

    Args:
        feature_weights: The weight of each feature in each component.
        sample_weights: The weight of each sample in each component.
        model: The fitted model.
        explained_variance_ratio: The proportion of variance explained by each component (e.g. for PCA).
    """

    feature_weights: pd.DataFrame
    sample_weights: pd.DataFrame
    model: Any
    explained_variance_ratio: pd.DataFrame | None = None


def _process_lvm_output(
    data: pd.DataFrame,
    component_names: list,
    feature_weights: np.ndarray,
    sample_weights: np.ndarray,
    model: Any,
    explained_variance_ratio: np.ndarray | None = None,
) -> LvmResults:
    feature_names = data.columns
    sample_names = data.index

    sample_weights = pd.DataFrame(
        data=sample_weights, columns=component_names, index=sample_names
    )

    feature_weights = pd.DataFrame(
        data=feature_weights, columns=feature_names, index=component_names
    )

    if explained_variance_ratio is not None:
        explained_variance_ratio = pd.DataFrame(
            {"component": component_names, "var": explained_variance_ratio}
        )

    return LvmResults(
        feature_weights=feature_weights,
        sample_weights=sample_weights,
        model=model,
        explained_variance_ratio=explained_variance_ratio,
    )


def pca(
    data: pd.DataFrame,
    n_components: int,
    scale: bool = True,
    random_state: int = 0,
    **kwargs: Any,
) -> LvmResults:
    """Performs principal component analysis (PCA).

    Before applying PCA, missing values in the data are replaced with
    the average of the corresponding feature and features are optionally
    standardised (zero mean and unit variance).

    Args:
        data: The input data.
        n_components : The number of principal components.
        scale : Whether the data will be scaled (mean = 0, sd = 1) or not.
        random_state: The random seed for reproducibility.
        **kwargs: Additional parameters to pass to the `PCA` function from sklearn.

    Returns:
        A NamedTuple containing the following elements:
        - feature_weights: The weights of each feature in each component.
        - sample_weights: The weights of each sample for each component.
        - model: The fitted  model object from sklearn.
        - explained_variance_ratio: The proportion of variance explained by each component.

    Examples:
        >>> import pandas as pd
        >>> from sklearn.datasets import load_iris
        >>> from omics_tools.multivariate import pca
        >>> data = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)
        >>> result = pca(data, n_components=2, scale=True)
        >>> print(result.feature_weights.head())
    """

    # Make a copy of the original data
    data_original = data.copy(deep=True)

    # Process data
    data = data.fillna(data.mean())
    if scale:
        data = StandardScaler().fit_transform(data)

    # Run model
    pca_model = PCA(n_components=n_components, random_state=random_state, **kwargs)
    sample_weights = pca_model.fit_transform(data)
    feature_weights = pca_model.components_

    return _process_lvm_output(
        data=data_original,
        component_names=[
            f"{n_components}_{random_state}_{index}" for index in range(n_components)
        ],
        sample_weights=sample_weights,
        feature_weights=feature_weights,
        explained_variance_ratio=pca_model.explained_variance_ratio_,
        model=pca_model,
    )


def ica(
    data: pd.DataFrame,
    n_components: int,
    features_as_sources: bool = True,
    random_state: int = 0,
    **kwargs: Any,
) -> LvmResults:
    """Performs Independent Component Analysis (ICA).

    Before applying ICA, missing values in the data are replaced with
    the average of the corresponding feature.

    Args:
        data: The input data.
        n_components: The number of independent components.
        features_as_sources: Indicates whether the features are interpreted as sources.
        random_state: The random seed for reproducibility.
        **kwargs: Additional parameters to pass to the `FastICA` function from sklearn.

    Returns:
        A NamedTuple containing the following elements.
        - feature_weights: The weights of each feature in each component.
        - sample_weights: The weights of each sample for each component.
        - model: The fitted  model object from sklearn.

    Examples:
        >>> import pandas as pd
        >>> from sklearn.datasets import load_iris
        >>> from omics_tools.multivariate import ica
        >>> data = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)
        >>> result = ica(data, n_components=2)
        >>> print(result.feature_weights.head())

    """

    # Make a copy of the original data
    data_original = data.copy(deep=True)

    # Process data
    data = data.fillna(data.mean())
    if features_as_sources:
        data = data.T

    # Run model
    ica_model = FastICA(n_components=n_components, random_state=random_state, **kwargs)
    sources = ica_model.fit_transform(data)
    mixing = ica_model.mixing_

    if features_as_sources:
        feature_weights = sources.T
        sample_weights = mixing
    else:
        feature_weights = mixing.T
        sample_weights = sources

    return _process_lvm_output(
        data=data_original,
        component_names=[
            f"{n_components}_{random_state}_{index}" for index in range(n_components)
        ],
        sample_weights=sample_weights,
        feature_weights=feature_weights,
        model=ica_model,
    )


def ica_reproducibility(
    data: pd.DataFrame, n_components: int, n_random_states: int = 10, **kwargs
) -> pd.DataFrame:
    """Evaluates the reproducibility of Independent Component Analysis (ICA).

    Runs ICA multiple times with different random states, keeping all
    other parameters constant. It then calculates the maximum correlation
    between each of the components of given run and all other components of
    another run.

    Args:
        data: The input data to perform ICA on.
        n_components: The number of components to decompose the data into.
        n_random_states: The number of different random states to use for ICA.
        **kwargs: Additional keyword arguments to pass to the ica function.

    Returns:
        A DataFrame containing the following columns:
        - n_components: the number of ICA components.
        - component_a: the component of run `a`.
        - random_state_a: the random state of run `a`.
        - random_state_b: the random state of of run `b`.
        - r2_max: the maximum correlation of `component_a` across all components of run `b`.

    Example:
        >>> import pandas as pd
        >>> from sklearn.datasets import load_iris
        >>> from omics_tools.multivariate import ica_reproducibility
        >>> data = pd.DataFrame(load_iris().data)
        >>> result = ica_reproducibility(data, n_components=3, n_random_states=5)
        >>> print(result.head())
    """

    results = pd.concat(
        [
            ica(
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


def _explained_variance_ratio_pls(
    pls_model: PLSRegression, dependent_variable: pd.Series
) -> np.ndarray:
    r2 = np.zeros(pls_model.n_components)
    for i in range(0, pls_model.n_components):
        dependent_variable_pred = (
            np.dot(
                pls_model.x_scores_[:, i][:, np.newaxis],
                pls_model.y_loadings_[:, i][:, np.newaxis].T,
            )
            * pls_model._y_std
            + pls_model._y_mean
        )
        r2[i] = r2_score(dependent_variable, dependent_variable_pred)

    return (
        r2 / r2.sum()
    )  # r2.sum() is equivalent to r2_score(dependent_variable, pls_model.predict(independent_variables))


def pls(
    independent_variables: pd.DataFrame,
    dependent_variable: pd.Series,
    n_components: int,
    **kwargs: Any,
) -> LvmResults:
    """Performs partial least squares regression (PLS).

    Before applying PLS, samples with missing values in the dependent
    variable are removed and missing values in a given independent variable
     are mean-imputed.

    Args:
        independent_variables: The independent (or predictor) variables
        depedent_variable: The dependent (or response) variable.
        n_components : The number of PLS components.
        **kwargs: Additional parameters to pass to the `PLSRegression` function from sklearn.

    Returns:
        A NamedTuple containing the following elements:
        - feature_weights: The weights of each feature in each component.
        - sample_weights: The weights of each sample for each component.
        - model: The fitted  model object from sklearn.
        - explained_variance_ratio: The proportion of variance explained by each component.

    Example:
    >>> import pandas as pd
    >>> from sklearn.datasets import load_diabetes
    >>> from omics_tools.multivariate import pls
    >>> diabetes = load_diabetes()
    >>> independent_variables = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    >>> dependent_variable = pd.Series(diabetes.target)
    >>> result = pls(independent_variables, dependent_variable, n_components=2)
    >>> print(result.feature_weights.head())

    """

    # Process data
    not_null_mask = ~dependent_variable.isnull()
    dependent_variable = dependent_variable[not_null_mask]
    independent_variables = independent_variables[not_null_mask]
    independent_variables = independent_variables.fillna(independent_variables.mean())

    # Run model
    pls_model = PLSRegression(n_components=n_components, **kwargs).fit(
        X=independent_variables, y=dependent_variable
    )

    return _process_lvm_output(
        data=independent_variables,
        component_names=[f"{n_components}_{index}" for index in range(n_components)],
        sample_weights=pls_model.x_scores_,
        feature_weights=pls_model.x_weights_.T,
        explained_variance_ratio=_explained_variance_ratio_pls(
            pls_model, dependent_variable
        ),
        model=pls_model,
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
            more than 2 different labels. Options: ‘micro’, ‘macro’, ‘samples’, ‘weighted’,
            ‘binary’ or None. For details see the `precision_score` function from sklearn.
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
