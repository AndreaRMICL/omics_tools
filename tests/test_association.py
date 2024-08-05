import numpy as np
import pandas as pd
import pytest
from scipy.stats import linregress, mannwhitneyu, ttest_ind

from omics_tools import association


@pytest.fixture()
def fixture_x_continuous() -> pd.DataFrame:
    continuous_variables = pd.DataFrame(
        {
            "feature1": [19, 22, 16, 29, 24, 20, 11, 17, 12],
            "feature2": [1, 2, 1, 2, np.nan, 2, 1, 2, 1],
        }
    )
    continuous_variables.index = [f"s{i}" for i in range(9)]
    return continuous_variables


@pytest.fixture()
def fixture_y_binary() -> pd.Series:
    return pd.Series(
        data=[1] * 5 + [0] * 4,
        index=[f"s{i}" for i in range(9)],
    )


@pytest.fixture()
def fixture_y_continuous() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature3": [10, 22, 16, 30, 24, 20, 11, 20, 12],
            "feature4": [1, 2, 3, 2, np.nan, 2, 4, 2, 1],
        },
        index=[f"s{i}" for i in range(9)],
    )


@pytest.mark.parametrize("method", list(association.AssociationMethodCategory))
@pytest.mark.parametrize("hypothesis", list(association.AlternativeHypothesis))
def test_compare_distributions(
    fixture_x_continuous, fixture_y_binary, method, hypothesis
):
    continuous_variables = fixture_x_continuous
    binary_variable = fixture_y_binary
    index_group = binary_variable[binary_variable == 1].index
    index_not_group = binary_variable[binary_variable == 0].index

    if method == association.AssociationMethodCategory.NON_PARAMETRIC:
        test_f = mannwhitneyu
    else:
        test_f = ttest_ind

    _, expected_pvalue_1 = test_f(
        continuous_variables.loc[index_group, "feature1"],
        continuous_variables.loc[index_not_group, "feature1"],
        alternative=hypothesis.value,
        nan_policy="omit",
    )
    _, expected_pvalue_2 = test_f(
        continuous_variables.loc[index_group, "feature2"],
        continuous_variables.loc[index_not_group, "feature2"],
        alternative=hypothesis.value,
        nan_policy="omit",
    )
    expected_pvalues = [expected_pvalue_1, expected_pvalue_2]

    observed_pvalues = association.compare_distributions(
        continuous_variables=continuous_variables,
        binary_variable=binary_variable,
        method_category=method,
        alternative_hypothesis=hypothesis,
    )["pvalue"].to_list()
    assert observed_pvalues == expected_pvalues


def test_correlation(fixture_x_continuous, fixture_y_continuous):
    continuous_variables_a = fixture_x_continuous
    continuous_variables_b = fixture_y_continuous

    shape_a = continuous_variables_a.shape
    shape_b = continuous_variables_b.shape
    n_expected_rows = shape_a[1] * shape_b[1]
    observed_all = association.correlation(
        continuous_variables_a=continuous_variables_a,
        continuous_variables_b=continuous_variables_b,
    )

    assert len(observed_all) == n_expected_rows


def test_linear_fit_parallel(fixture_x_continuous, fixture_y_continuous):
    independent_variables = fixture_x_continuous.to_numpy(dtype=float)
    dependent_variables = fixture_y_continuous.to_numpy(dtype=float)
    independent_variables[np.isnan(independent_variables)] = (
        0  # linregress from scipy.stats cannot handle NaNs
    )
    dependent_variables[np.isnan(dependent_variables)] = 0
    covariates = np.empty((len(independent_variables), 0))

    n_features_y = dependent_variables.shape[1]
    n_features_x = independent_variables.shape[1]

    observed_intercept, observed_beta, observed_se, observed_t, observed_r, _, _ = (
        association.linear_fit_parallel(
            independent_variables=independent_variables,
            dependent_variables=dependent_variables,
            covariates=covariates,
        )
    )

    # Check results using individual linear regressions for comparison
    expected_beta = np.empty((n_features_y, n_features_x))
    expected_intercept = np.empty((n_features_y, n_features_x))
    expected_se = np.empty((n_features_y, n_features_x))
    expected_r = np.empty((n_features_y, n_features_x))
    for i in range(n_features_y):
        for j in range(n_features_x):
            (
                expected_beta[i, j],
                expected_intercept[i, j],
                expected_r[i, j],
                _,
                expected_se[i, j],
            ) = linregress(independent_variables[:, j], dependent_variables[:, i])
    np.testing.assert_array_equal(
        np.round(observed_intercept, 3), np.round(expected_intercept, 3)
    )
    np.testing.assert_array_equal(
        np.round(observed_beta, 3), np.round(expected_beta, 3)
    )
    np.testing.assert_array_equal(np.round(observed_r, 3), np.round(expected_r**2, 3))
    np.testing.assert_array_equal(np.round(observed_se, 3), np.round(expected_se, 3))


def test_fast_linear_regression(fixture_x_continuous, fixture_y_continuous):
    independent_variables = fixture_x_continuous
    dependent_variables = fixture_y_continuous

    results = association.fast_linear_regression(
        independent_variables=independent_variables,
        dependent_variables=dependent_variables,
    )

    assert len(results) == independent_variables.shape[1] * dependent_variables.shape[1]


def test_regression_linear(fixture_x_continuous, fixture_y_continuous):
    independent_variables = fixture_x_continuous
    dependent_variable = fixture_y_continuous["feature3"]

    results = association.regression(
        independent_variables=independent_variables[["feature1"]],
        dependent_variable=dependent_variable,
        regression_method=association.RegressionMethod.LINEAR,
    )
    expected_beta, _, _, expected_pvalue, expected_se = linregress(
        independent_variables["feature1"], dependent_variable
    )

    assert np.round(results["beta"][0], 3) == np.round(expected_beta, 3)
    assert np.round(results["se"][0], 3) == np.round(expected_se, 3)
    assert np.round(results["pvalue"][0], 3) == np.round(expected_pvalue, 3)


def test_regression_logistic(
    fixture_x_continuous, fixture_y_continuous, fixture_y_binary
):
    independent_variables = fixture_x_continuous
    covariates = fixture_y_continuous
    dependent_variable = fixture_y_binary

    results = association.regression(
        independent_variables=independent_variables,
        dependent_variable=dependent_variable,
        covariates=covariates,
        regression_method=association.RegressionMethod.LOGISTIC,
    )

    assert len(results) <= independent_variables.shape[1]
