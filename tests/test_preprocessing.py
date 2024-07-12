import numpy as np
import pandas as pd
import pytest

from omics_tools import preprocessing


@pytest.fixture()
def fixture_data() -> pd.DataFrame:
    return pd.DataFrame({"A": [1, 2, 3, np.nan, 4], "C": [1, 2, 3, 4, 5]})


def test_inverse_rank_transformation(fixture_data):
    data = fixture_data

    observed = preprocessing.inverse_normal_transformation(data=data).round(2)

    expected = pd.DataFrame(
        {"A": [-1.15, -0.32, 0.32, np.nan, 1.15], "C": [-1.28, -0.52, 0.00, 0.52, 1.28]}
    )

    pd.testing.assert_frame_equal(observed, expected)


def test_quantile_transform(fixture_data):
    data = fixture_data

    observed = preprocessing.quantile_transformation(data=data).round(2)

    expected = pd.DataFrame(
        {"A": [0.00, 0.33, 0.67, np.nan, 1.0], "C": [0, 0.25, 0.5, 0.75, 1]}
    )

    pd.testing.assert_frame_equal(observed, expected)
