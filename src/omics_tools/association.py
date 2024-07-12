import itertools
import warnings
from enum import Enum, unique

import numba
import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu, pearsonr, spearmanr, t, ttest_ind
from statsmodels.stats.multitest import multipletests


@unique
class AssociationMethodCategory(Enum):
    PARAMETRIC = "PARAMETRIC"
    NON_PARAMETRIC = "NON_PARAMETRIC"


@unique
class AlternativeHypothesis(Enum):
    TWO_SIDED = "two-sided"
    LESS = "less"
    GREATER = "greater"


@unique
class RegressionMethod(Enum):
    LINEAR = "LINEAR"
    LOGISTIC = "LOGISTIC"


def compare_distributions(
    continuous_variables: pd.DataFrame,
    binary_variable: pd.Series,
    method_category: AssociationMethodCategory = AssociationMethodCategory.NON_PARAMETRIC,
    alternative_hypothesis: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
    min_group_size: int = 2,
) -> pd.DataFrame:
    """Compares the distribution of values in two groups.

    Compares the feature distribution between two groups (defined by the binary variable)
    using either parametric (t-test) or non-parametric (Mann-Whitney U test) approaches.

    Args:
        continuous_variables: The DataFrame of continuous variables where each
            row is a sample and each column is a feature.
        binary_variable: A binary Series indicating group membership, with values
            0 or 1, and same index as `continuous variables`.
        method_category: The category of statistical method to use for testing
            associations. Either `AssociationMethodCategory.PARAMETRIC` or
            `AssociationMethodCategory.NON_PARAMETRIC`.
        alternative_hypothesis: The alternative hypothesis for the statistical test.
            Either `AlternativeHypothesis.TWO_SIDED`, `AlternativeHypothesis.GREATER`,
            or `AlternativeHypothesis.LESS`.
        min_group_size: The minimum size of the two groups under comparison.
    Returns:
        A DataFrame containing the following columns:
        - `feature`: The feature names.
        - `avg_group`: The average value of each feature in the group where `y` is 1.
        - `avg_not_group`: The average value of each feature in the group where `y` is 0.
        - `avg_group_ratio`: The ratio of `avg_group` to `avg_not_group`.
        - `pvalue`: The p-values from the statistical tests.
        - `pfdr`: The adjusted p-values using the FDR method (Benjamini/Hochberg).

    Raises:
        ValueError: If an invalid `method_category` is provided.

    Warnings:
        UserWarning: If either group under comparison has fewer than 2 samples,
        the function skips the test and returns an empty DataFrame.

    Examples:
        >>> import pandas as pd
        >>> from omics_tools.association import compare_distributions
        >>> continuous_variables = pd.DataFrame({
        >>>     "feature1": [1.2, 3.4, 2.1, 4.5, 3.3],
        >>>     "feature2": [5.5, 2.6, 3.7, 4.1, 5.0],
        >>> }, index = ["s1", "s2", "s3", "s4", "s5"])
        >>> binary_variable = pd.Series([0, 1, 1, 0, 1], index = ["s1", "s2", "s3", "s4", "s5"])
        >>> results = compare_distributions(continuous_variables, binary_variable)
        >>> print(results)
    """

    index_group = set(binary_variable[binary_variable == 1].index) & set(
        continuous_variables.index
    )
    index_not_group = set(binary_variable[binary_variable == 0].index) & set(
        continuous_variables.index
    )
    if (len(index_group) < min_group_size) | (len(index_not_group) < min_group_size):
        warnings.warn(
            "Skipping test: the groups under comparison are too small", UserWarning
        )
        return pd.DataFrame({})

    data_group = continuous_variables.loc[list(index_group)]
    data_not_group = continuous_variables.loc[list(index_not_group)]

    if method_category == AssociationMethodCategory.PARAMETRIC:
        agg_f = np.nanmean
        test_f = ttest_ind
    elif method_category == AssociationMethodCategory.NON_PARAMETRIC:
        agg_f = np.nanmedian
        test_f = mannwhitneyu
    else:
        raise ValueError("Invalid method")

    averages_group = data_group.apply(agg_f)
    averages_not_group = data_not_group.apply(agg_f)
    averages_not_group = averages_not_group.loc[averages_group.index]
    _, pvalues = test_f(
        data_group.to_numpy(),
        data_not_group.to_numpy(),
        alternative=alternative_hypothesis.value,
        nan_policy="omit",
    )

    return pd.DataFrame(
        {
            "feature": averages_group.index,
            "avg_group": averages_group,
            "avg_not_group": averages_not_group,
            "avg_group_ratio": averages_group / averages_not_group,
            "pvalue": pvalues,
            "pfdr": multipletests(np.nan_to_num(pvalues, nan=1), method="fdr_bh")[1],
        }
    ).reset_index(drop=True)


def correlation(
    continuous_variables_a: pd.DataFrame,
    continuous_variables_b: pd.DataFrame,
    method_category: AssociationMethodCategory = AssociationMethodCategory.NON_PARAMETRIC,
) -> pd.DataFrame:
    """Computes correlations.

    Computes Pearson or Spearman correlation coefficients and p-values between each
    pair of features from the dataframes `continuous_variables_a` and `continuous_variables_n`.

    Args:
        continuous_variables_a: A DataFrame where each column is a feature.
        continuous_variables_b: A DataFrame where each column is a feature.
        method_category: The category of statistical method to use for testing
            associations. Either `AssociationMethodCategory.PARAMETRIC`for Pearson's
            correlation or`AssociationMethodCategory.NON_PARAMETRIC` for Spearman's
            correlation.

    Returns:
        A DataFrame with the following columns:
        - 'feature_a': The feature names from continuous_variables_a.
        - 'feature_b': The feature names from continuous_variables_b.
        - 'r': Spearman correlation coefficient for the feature pair.
        - 'pvalue': The p-value for the correlation coefficient.
        - pfdr: The adjusted p-values using the FDR method (Benjamini/Hochberg).
        The adjustment is done for each feature_b separately.


    Examples:
        >>> import pandas as pd
        >>> from omics_tools.association import correlation
        >>> continuous_variables_a = pd.DataFrame({
        >>>     'feature1': [1, 2, 3],
        >>>     'feature2': [4, 5, 6]
        >>> })
        >>> continuous_variables_b = pd.DataFrame({
        >>>     'feature3': [8, 8, 1],
        >>>     'feature4': [1, 0, 2]
        >>> })
        >>> correlation(continuous_variables_a, continuous_variables_b)
    """

    if method_category == AssociationMethodCategory.PARAMETRIC:
        corr_f = pearsonr
    elif method_category == AssociationMethodCategory.NON_PARAMETRIC:
        corr_f = spearmanr
    else:
        raise ValueError("Invalid method")

    combinations = list(
        itertools.product(
            continuous_variables_a.columns, continuous_variables_b.columns
        )
    )

    results = []

    for feature_a, feature_b in combinations:
        tmp = continuous_variables_a[[feature_a]].join(
            continuous_variables_b[[feature_b]]
        )
        tmp = tmp[~tmp.isnull().any(axis=1)]
        r, pvalue = corr_f(tmp[feature_a], tmp[feature_b])
        results.append(
            pd.DataFrame(
                {
                    "feature_a": [feature_a],
                    "feature_b": [feature_b],
                    "r": [r],
                    "pvalue": [pvalue],
                }
            )
        )

    results = pd.concat(results).sort_values("feature_b")
    results["pfdr"] = (
        results.groupby("feature_b", sort=False)
        .apply(
            lambda x: multipletests(x["pvalue"].fillna(1), method="fdr_bh")[1],
            include_groups=False,
        )
        .explode()
        .to_list()
    )

    return results


signatures = "UniTuple(f8[:,:],7)(f8[:,:], f8[:,:], f8[:,:])"


@numba.njit(signatures, nogil=True, parallel=True, cache=True)
def linear_fit_parallel(
    independent_variables: np.ndarray,
    dependent_variables: np.ndarray,
    covariates: np.ndarray,
):
    """Performs linear regression fits for multiple datasets in parallel.

    Args:
        independent_variables: A 2D numpy array of shape (n_samples, n_features_x)
            containing the independent variable data.
        dependent_variables: A 2D numpy array of shape (n_samples, n_features_y)
            containing the dependent variable data.
        covariates: A 2D numpy array of shape (n_samples, n_covariates) containing
            the covariate data. If no covariates, an empty array with shape (n_samples, 0)
            should be passed.

    Returns:
        A tuple containing the following 2D numpy arrays, each of shape (n_features_y, n_features_x):
            - intercept: The intercepts of the linear regression models.
            - beta: The beta coefficients of the linear regression models.
            - se: The standard errors of the coefficients.
            - t: The t-statistics.
            - r2: The coefficients of determination.
            - size: The sample sizes.
            - dof: The degrees of freedom.

    Raises:
        AssertionError: If the number of samples in XX, YY, and CC do not match.

    Examples:
        >>> import numpy as np
        >>> from omics_tools.association import linear_fit_parallel
        >>> independent_variables = np.random.rand(100, 10)
        >>> dependent_variables = np.random.rand(100, 5)
        >>> covariates = np.random.rand(100, 3)
        >>> intercepts, beta, se, t, r2, size, dof = linear_fit_parallel(independent_variables, dependent_variables, covariates)
    """

    n_samples_x, n_pnts_x = independent_variables.shape
    n_samples_y, n_pnts_y = dependent_variables.shape
    n_samples_c, n_pnts_c = covariates.shape

    if n_samples_x != n_samples_y:
        raise ValueError(
            "Independent_variables and dependent_variables have different numbers of samples"
        )
    if n_samples_x != n_samples_c:
        raise ValueError(
            "Independent_variables and covariates have different numbers of samples"
        )

    intercept = np.empty((n_pnts_y, n_pnts_x))  # matrix of intercepts
    beta = np.empty((n_pnts_y, n_pnts_x))  # matrix of beta coefficients
    se = np.empty((n_pnts_y, n_pnts_x))  # matrix of standard errors
    t = np.empty((n_pnts_y, n_pnts_x))  # matrix of t-statistics
    r2 = np.empty((n_pnts_y, n_pnts_x))  # matrix of R-squared values
    size = np.empty((n_pnts_y, n_pnts_x))  # matrix of sample sizes
    dof = np.empty((n_pnts_y, n_pnts_x))  # matrix of degrees of freedom

    for j in numba.prange(n_pnts_x):  # for each column in XX
        # Get column j
        X = independent_variables[:, j]  # (n_samples_x, 1)

        # Add covariates
        XC = np.column_stack((X, covariates))  # (n_samples_x, 1 + n_pnts_c)

        # Clean NaNs
        # where any of XC or Y is NA, remove NA values from XC and YY
        mask = (np.isnan(XC).sum(axis=1) == 0) & (
            np.isnan(dependent_variables).sum(axis=1) == 0
        )
        XC_clean = XC[mask]  # (n_samples_clean, 1 + n_pnts_c)
        Y_clean = dependent_variables[mask]  # (n_samples_clean, n_pnts_y)
        n_samples_clean = XC_clean.shape[0]

        # Add intercepts to XC
        A = np.column_stack(
            (np.ones(n_samples_clean), XC_clean)
        )  # (n_samples_clean, 2 + n_pnts_c)
        n_params = A.shape[1]

        # Fit X to all YY
        PARAMS, RES, _, _ = np.linalg.lstsq(A, Y_clean)

        # Get intercepts
        INTERCEPTS = PARAMS[0]
        intercept[:, j] = INTERCEPTS

        # Get beta coefficients
        BETAS = PARAMS[1]  # beta coefficients for this X and all YY
        beta[:, j] = BETAS

        # Calculate number of degrees of freedom for all fits
        ndof = n_samples_clean - n_params  # number of dof
        size[:, j] = n_samples_clean
        dof[:, j] = ndof

        # Get standard error of model parameters
        MSE = RES / ndof  # (n_pnts_y, )
        C = np.linalg.pinv(np.dot(A.T, A))  # (n_params, n_params)
        dC = np.diag(C)  # (n_params, )
        VARS = np.outer(dC, MSE)  # (n_params, n_pnts_y)
        SE_BETA = np.sqrt(VARS[1])  # standard error on beta; (n_pnts_y, )
        se[:, j] = SE_BETA

        # Get t-statistic
        t[:, j] = BETAS / SE_BETA

        # Calculate R-squared
        y_mean = Y_clean.sum(axis=0) / len(Y_clean)
        SST = ((Y_clean - y_mean) ** 2).sum(axis=0)
        r2[:, j] = 1 - (RES / SST)

    # all outputs have shape (n_pnts_y, n_pns_x)
    return intercept, beta, se, t, r2, size, dof


def fast_linear_regression(
    independent_variables: pd.DataFrame,
    dependent_variables: pd.DataFrame,
    covariates: pd.DataFrame = None,
) -> pd.DataFrame:
    """Performs fast linear regression for multiple datasets.

    Performs linear regressions for each feature in `independent_variables` and each
    feature in `dependent_variables`, optionally adjusting for covariates.

    Args:
        independent_variables: The DataFrame containing the independent variables.
        dependent_variables: The DataFrame containing the dependent variables.
        covariates: The DataFrame containing the covariate data.

    Returns:
        The DataFrame containing the regression results, with columns:
            - feature_x: The independent variables.
            - feature_y: The dependent variables.
            - beta: The beta coefficients.
            - se: The standard errors of the beta coefficients.
            - statistic: The t-statistics.
            - pvalue: The p-values for the t-statistics.
            - pfdr: The adjusted p-values using the FDR method (Benjamini/Hochberg).

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from omics_tools.association import fast_linear_regression
        >>> n_samples = 100
        >>> independent_variables = pd.DataFrame(np.random.randn(n_samples, 5))
        >>> dependent_variables = pd.DataFrame(np.random.randn(n_samples, 10))
        >>> covariates = pd.DataFrame(np.random.randn(n_samples, 2))
        >>> results = fast_linear_regression(independent_variables, dependent_variables, covariates)
        >>> print(results)

    """

    intercept, beta, se, statistic, r2, size, dof = linear_fit_parallel(
        independent_variables=independent_variables.to_numpy(dtype=float),
        dependent_variables=dependent_variables.to_numpy(dtype=float),
        covariates=(
            covariates.to_numpy(dtype=float)
            if covariates is not None
            else np.empty((independent_variables.shape[0], 0))
        ),
    )

    # Create feature names
    feature_x_names = independent_variables.columns
    feature_y_names = dependent_variables.columns

    # Get the number of features
    n_features_x = len(feature_x_names)
    n_features_y = len(feature_y_names)

    # Repeat feature names to match the dimensions of the dataframe of results
    feature_x_repeated = np.tile(feature_x_names, n_features_y)
    feature_y_repeated = np.repeat(feature_y_names, n_features_x)

    results = pd.DataFrame(
        {
            "feature_x": feature_x_repeated,
            "feature_y": feature_y_repeated,
            "beta": beta.ravel(),
            "se": se.ravel(),
            "statistic": statistic.ravel(),
            "pvalue": 2
            * t.sf(np.abs(statistic.ravel()), dof.ravel()),  # two-sided pvalue
        }
    )
    results["pfdr"] = (
        results.groupby("feature_y", sort=False)
        .apply(
            lambda x: multipletests(x["pvalue"].fillna(1), method="fdr_bh")[1],
            include_groups=False,
        )
        .explode()
        .to_list()
    )

    return results


def _regression_single(
    independent_variable: pd.Series,
    dependent_variable: pd.Series,
    covariates: pd.DataFrame = None,
    regression_method: RegressionMethod = RegressionMethod.LINEAR,
    independent_variable_name: str = "x",
    dependent_variable_name: str = "y",
) -> pd.DataFrame:
    """Performs linear or logistic regression.

    Performs linear or logistic regression for a pair of independent and dependent
    variables, with optional adjustment for covariates.

    Args:
        independent_variable: The independent variable.
        dependent_variable: The dependent variable.
        covariates: The covariate data. Defaults to None.
        regression_method: The regression method of regression to perform. Either
            RegressionMethod.LINEAR or RegressionMethod.LOGISTIC.
        independent_variable_name: The name of the independent variable.
        dependent_variable_name: The name of the dependent variable.

    Returns:
        The DataFrame containing the regression results, with columns:
            - feature_x: The independent variables.
            - feature_y: The dependent variables.
            - beta: The beta coefficient.
            - se: The standard error of the beta coefficient.
            - pvalue: The p-value.

    Example:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from omics_tools.association import _regression_single
        >>> n_samples = 100
        >>> independent_variables = pd.DataFrame(np.random.randn(n_samples, 2))
        >>> dependent_variables = pd.DataFrame(np.random.randn(n_samples, 2))
        >>> covariates = pd.DataFrame(np.random.randn(n_samples, 2))
        >>> results = _regression_single(independent_variables[0], dependent_variables[0], covariates)
        >>> print(results)
    """

    try:
        # Combine x, y, and c into a single DataFrame and drop NaNs
        data = pd.concat(
            [dependent_variable, independent_variable, covariates], axis=1
        ).dropna()

        # Extract cleaned x, y, and c
        X = data.iloc[:, 1:]
        y_clean = data.iloc[:, 0]
        X["const"] = 1

        if regression_method == RegressionMethod.LINEAR:
            model = sm.OLS(y_clean, X)
        elif regression_method == RegressionMethod.LOGISTIC:
            model = sm.Logit(y_clean, X)
        else:
            raise ValueError("Invalid method.")

        model_results = model.fit(disp=0).summary2().tables[1]
        model_results = model_results.iloc[[0]]
        pval_column = [column for column in model_results.columns if "P" in column][0]

        return pd.DataFrame(
            {
                "feature_x": independent_variable_name,
                "feature_y": dependent_variable_name,
                "beta": model_results["Coef."],
                "se": model_results["Std.Err."],
                "pvalue": model_results[pval_column],
            },
            columns=["feature_y", "feature_x", "beta", "se", "pvalue"],
        ).reset_index(drop=True)

        return results

    except Exception as e:
        warnings.warn(
            f"Regression failed for {independent_variable_name} and {dependent_variable_name}: {e}"
        )
        return pd.DataFrame({})


def regression(
    independent_variables: pd.DataFrame,
    dependent_variable: pd.Series,
    covariates: pd.DataFrame = None,
    regression_method: RegressionMethod = RegressionMethod.LINEAR,
) -> pd.DataFrame:
    """Performs regression analysis.

    Performs a linear a logistic regression between the `dependent variable` and
    each feature in `independent_variables`, optionally adjusting for covariates.

    Args:
        independent_variables: The DataFrame containing the independent variables.
        dependent_variable: The dependent variable.
        covariates: The DataFrame containing the covariate data.
        method: The regression method of regression to perform. Either
        RegressionMethod.LINEAR or RegressionMethod.LOGISTIC.

    Returns:
        The DataFrame containing the regression results, with columns:
            - feature_x: The independent variables.
            - feature_y: The dependent variables.
            - beta: The beta coefficients.
            - se: The standard errors of the beta coefficients.
            - pvalue: The p-values.
            - pfdr: The adjusted p-values using the FDR method (Benjamini/Hochberg).

    Example:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from omics_tools.association import regression
        >>> n_samples = 100
        >>> independent_variables = pd.DataFrame(np.random.randn(n_samples, 2))
        >>> dependent_variables = pd.DataFrame(np.random.randn(n_samples, 2))
        >>> covariates = pd.DataFrame(np.random.randn(n_samples, 2))
        >>> results = regression(independent_variables, dependent_variables[0], covariates)
        >>> print(results)
    """

    results = Parallel(n_jobs=-1)(
        delayed(_regression_single)(
            independent_variable=independent_variables[col],
            dependent_variable=dependent_variable,
            covariates=covariates,
            independent_variable_name=col,
            regression_method=regression_method,
        )
        for col in independent_variables.columns
    )
    results = pd.concat(results)
    if not results.empty:
        results["pfdr"] = multipletests(results["pvalue"].fillna(1), method="fdr_bh")[1]

    return results
