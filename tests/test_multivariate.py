import pandas as pd
import pytest
from sklearn.datasets import load_diabetes

from omics_tools import multivariate


@pytest.fixture()
def fixture_dataset() -> pd.DataFrame:
    diabetes = load_diabetes()
    data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    target = pd.Series(diabetes.target)

    return data, target


def test_pca(fixture_dataset):
    data = fixture_dataset[0]

    pca_results = multivariate.pca(data=data, n_components=2)

    assert set(pca_results.feature_weights.columns) == set(data.columns)
    assert set(pca_results.sample_weights.index) == set(data.index)


def test_ica(fixture_dataset):
    data = fixture_dataset[0]

    ica_results = multivariate.ica(data=data, n_components=2)

    assert set(ica_results.feature_weights.columns) == set(data.columns)
    assert set(ica_results.sample_weights.index) == set(data.index)


def test_pls(fixture_dataset):
    data = fixture_dataset[0]
    dependent_variable = fixture_dataset[1]

    pls_results = multivariate.pls(
        independent_variables=data,
        dependent_variable=dependent_variable,
        n_components=3,
    )

    assert set(pls_results.feature_weights.columns) == set(data.columns)
    assert set(pls_results.sample_weights.index) == set(data.index)
