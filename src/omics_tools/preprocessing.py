import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import quantile_transform


def _int(values: pd.Series, ties_method="average") -> pd.Series:
    """Performs inverse normal transformation (INT).

    Args:
        values: A pandas Series containing numeric values to transform.
        ties_method: The method used to handle ties when ranking.
            See `pandas.Series.rank()` documentation for available options.

    Returns:
        The transformed Series where non-null values are replaced with their
        corresponding inverse normal transformed values.
    """

    # Get a mask of non-null values
    not_nan_mask = ~values.isnull()
    values_not_nan = values[not_nan_mask]

    # Apply INT
    transformed_values_not_nan = norm.ppf(
        (values_not_nan.rank(method=ties_method) - 0.5) / len(values_not_nan)
    )
    values[not_nan_mask] = transformed_values_not_nan

    return values


def inverse_normal_transformation(
    data: pd.DataFrame, ties_method="average"
) -> pd.DataFrame:
    """Performs rank-based inverse normal transformation (INT).

    Args:
        data: A pandas DataFrame containing numeric columns to transform.
        ties_method: Method used to handle ties when ranking. See pandas.Series.rank()
            documentation for available options.

    Returns:
        Transformed pandas DataFrame where each column has been transformed using
        inverse normal transformation.

    Example:
        >>> import pandas as pd
        >>> from sklearn.datasets import load_iris
        >>> from omics_tools.preprocessing import inverse_normal_transformation
        >>> data = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)
        >>> transformed_data = inverse_normal_transformation(data)
        >>> print(transformed_data)
    """

    data_transformed = data.copy(deep=True).apply(_int, axis=0, ties_method=ties_method)

    return data_transformed


def quantile_transformation(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Applies quantile-based transformation.

    Args:
        data: The DataFrame containing numeric columns to transform.
        **kwargs: Additional keyword arguments to pass to `quantile_transform` function
            from sklearn.

    Returns:
        The transformed DataFrame where each numeric column has been quantile transformed.

    Example:
        >>> import pandas as pd
        >>> from sklearn.datasets import load_iris
        >>> data = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)
        >>> transformed_data = quantile_transformation(data, n_quantiles=5, random_state=0)
        >>> print(transformed_data)
    """
    return pd.DataFrame(
        {
            col: quantile_transform(data[col].values.reshape(-1, 1), **kwargs).flatten()
            for col in data.columns
        },
        index=data.index,
    )
