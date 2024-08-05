import pandas as pd
import pytest
from sklearn.datasets import load_iris

from omics_tools import evaluation

@pytest.fixture()
def fixture_dataset() -> pd.DataFrame:
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)

    return data



def test_ica_reproducibility(fixture_dataset):
    data = fixture_dataset

    n_components = 4
    n_random_states = 6
    results = evaluation.lvm_reproducibility(
        data=data,
        n_components=n_components,
        n_random_states=n_random_states,
        lvm_method=evaluation.LvmMethod.ICA,
    )

    expected_length = (n_components * (n_random_states - 1)) * n_random_states
    observed_length = len(results)

    assert expected_length == observed_length


def test_pca_reproducibility(fixture_dataset):
    data = fixture_dataset

    n_components = 4
    n_random_states = 6
    results = evaluation.lvm_reproducibility(
        data=data,
        n_components=n_components,
        n_random_states=n_random_states,
        lvm_method=evaluation.LvmMethod.PCA,
    )

    expected_length = (n_components * (n_random_states - 1)) * n_random_states
    observed_length = len(results)

    assert expected_length == observed_length


def test_nmf_reproducibility(fixture_dataset):
    data = fixture_dataset

    n_components = 4
    n_random_states = 6
    results = evaluation.lvm_reproducibility(
        data=data,
        n_components=n_components,
        n_random_states=n_random_states,
        lvm_method=evaluation.LvmMethod.NMF,
    )

    expected_length = (n_components * (n_random_states - 1)) * n_random_states
    observed_length = len(results)

    assert expected_length == observed_length
