import pandas as pd
import pytest

from omics_tools import tissue_specificity


@pytest.fixture()
def fixture_tissue_expression() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gene_name": ["GFAP"] * 3 + ["INS"] * 3,
            "tissue": ["brain", "liver", "pancreas"] * 2,
            "tpm": [20, 1, 1, 1, 10, 1],
        }
    )


def test_hpa_enhancement(fixture_tissue_expression):
    tissue_expression = fixture_tissue_expression
    observed = tissue_specificity.hpa(
        tissue_expression=tissue_expression,
        gene_column="gene_name",
        expression_column="tpm",
        method=tissue_specificity.HpaMethod.ENHANCEMENT,
    )
    observed["enhancement"] = observed["enhancement"].round(3)

    expected = tissue_expression.assign(
        enhancement=[20.000, 0.095, 0.095, 0.182, 10.000, 0.182]
    )

    pd.testing.assert_frame_equal(
        observed.sort_values(["gene_name", "tissue"]), expected
    )


def test_hpa_enrichment(fixture_tissue_expression):
    tissue_expression = fixture_tissue_expression
    observed = tissue_specificity.hpa(
        tissue_expression=tissue_expression,
        gene_column="gene_name",
        expression_column="tpm",
        method=tissue_specificity.HpaMethod.ENRICHMENT,
    )
    observed["enrichment"] = observed["enrichment"].round(3)

    expected = pd.DataFrame(
        {
            "gene_name": ["GFAP", "INS"],
            "tissue": ["brain", "liver"],
            "tpm": [20, 10],
            "enrichment": [20.0, 10.0],
        }
    )

    pd.testing.assert_frame_equal(
        observed.sort_values(["gene_name", "tissue"]), expected
    )


def test_pem(fixture_tissue_expression):
    tissue_expression = fixture_tissue_expression
    observed = tissue_specificity.pem(
        tissue_expression=tissue_expression,
        gene_column="gene_name",
        expression_column="tpm",
        tissue_column="tissue",
    )
    observed["pem"] = observed["pem"].round(3)
    expected = tissue_expression.assign(
        pem=[0.168, -0.852, -0.112, -0.870, 0.411, 0.151]
    )
    pd.testing.assert_frame_equal(
        observed.sort_values(["gene_name", "tissue"]), expected
    )
