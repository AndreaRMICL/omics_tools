from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import NMF, PCA, FastICA
from sklearn.metrics import r2_score
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


def nmf(
    data: pd.DataFrame,
    n_components: int,
    random_state: int = 0,
    **kwargs: Any,
) -> LvmResults:
    """Performs non-negative matrix factorisation (NMF).

    Before applying NMF, missing values in the data are replaced with
    the average of the corresponding feature.

    Args:
        data: The input data.
        n_components : The number of principal components.
        random_state: The random seed for reproducibility.
        **kwargs: Additional parameters to pass to the `PCA` function from sklearn.

    Returns:
        A NamedTuple containing the following elements:
        - feature_weights: The weights of each feature in each component.
        - sample_weights: The weights of each sample for each component.
        - model: The fitted  model object from sklearn.

    Examples:
        >>> import pandas as pd
        >>> from sklearn.datasets import load_iris
        >>> from omics_tools.multivariate import nmf
        >>> data = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)
        >>> result = nmf(data, n_components=2, scale=True)
        >>> print(result.feature_weights.head())
    """

    # Make a copy of the original data
    data_original = data.copy(deep=True)

    # Process data
    data = data.fillna(data.mean())

    # Run model
    nmf_model = NMF(n_components=n_components, random_state=random_state, **kwargs)
    sample_weights = nmf_model.fit_transform(data)
    feature_weights = nmf_model.components_

    return _process_lvm_output(
        data=data_original,
        component_names=[
            f"{n_components}_{random_state}_{index}" for index in range(n_components)
        ],
        sample_weights=sample_weights,
        feature_weights=feature_weights,
        model=nmf_model,
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
