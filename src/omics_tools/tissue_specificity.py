from enum import Enum, unique

import numpy as np
import pandas as pd


@unique
class HpaMethod(Enum):
    ENHANCEMENT = "ENHANCEMENT"
    ENRICHMENT = "ENRICHMENT"


def _tissue_enhancement(
    tissue_expression: pd.DataFrame, expression_column: str
) -> pd.DataFrame:
    """Calculates tissue enhancement.

    Calculates tissue enhancement as defined by the Human Protein Atlas (HPA).
    Compares the expression of a given gene in a given tissue to the mean expression
    of that gene in all other tissues.

    Args:
        tissue_expression: The DataFrame of containing the expression values
            of a given gene across different tissues.
        expression_column: The column containing the gene expression values.

    Returns:
        The input DataFrame with an added 'enhancement' column.
    """
    mean_values_without_self = (
        tissue_expression[expression_column].sum()
        - tissue_expression[expression_column]
    ) / (len(tissue_expression) - 1)
    tissue_expression["enhancement"] = (
        tissue_expression[expression_column] / mean_values_without_self
    )

    return tissue_expression


def _tissue_enrichment(
    tissue_expression: pd.DataFrame, expression_column: str
) -> pd.DataFrame:
    """Calculates tissue enrichment.

    Calculates tissue enrichment as defined by the Human Protein Atlas (HPA).
    Compares the highest expression of a given gene across all tissues to the
    expression of that gene in any other tissue.

    Args:
        tissue_expression: The DataFrame of containing the expression values
            of a given gene across different tissues.
        expression_column: The column containing the gene expression values.

    Returns:
        The input DataFrame with an added 'enrichment' column.
    """
    tissue_expression = tissue_expression.sort_values(
        expression_column, ascending=False
    )
    gene_expression_values = tissue_expression[expression_column].to_list()
    tissue_expression["enrichment"] = (
        gene_expression_values[0] / gene_expression_values[1]
    )

    return tissue_expression.head(1)


def hpa(
    tissue_expression: pd.DataFrame,
    gene_column: str,
    expression_column: str,
    method: HpaMethod,
) -> pd.DataFrame:
    """Calculates tissue specificity using the HPA definitions.

    Calculates tissue specificity as defined by the Human Protein Atlas (HPA) [1]_.
    It supports two different metrics:
        - Tissue enhancement: compares the expression of a each gene in a given tissue
        to the mean expression of that gene in all other tissues.
        - Tissue enrichment: compares the highest expression of a given gene across
        all tissues to the expression of that gene in any other tissue.

    Args:
        tissue_expression: The DataFrame of containing the gene expression data
            across different tissues.
        gene_column: The column containing gene identifiers.
        expression_column: The column containing the gene expression values.
        method: The tissue specificty method. Either `HpaMethod.ENHANCEMENT` or
            `HpaMethod.ENRICHMENT`.

    Returns:
        The input DataFrame with an added column reporting tissue specificity.

    References:
        [1] https://www.proteinatlas.org/humanproteome/tissue/tissue+specific.

    Example:
        >>> import pandas as pd
        >>> from omics_tools.tissue_specificity import hpa, HpaMethod
        >>> tissue_expression = pd.DataFrame({
        >>>     "gene": ["GFAP"] * 3 + ["INS"] * 3,
        >>>     "tissue": ["brain", "liver", "pancreas"] * 2,
        >>>     "expression": [20, 1, 1, 1, 10, 1],
        >>> })
        >>> result = hpa(tissue_expression, gene_column="gene", expression_column="expression", method=HpaMethod.ENHANCEMENT)
        >>> print(result)
        >>> result = hpa(tissue_expression, gene_column="gene", expression_column="expression", method=HpaMethod.ENRICHMENT)
        >>> print(result)
    """

    if method == HpaMethod.ENHANCEMENT:
        f = _tissue_enhancement
    elif method == HpaMethod.ENRICHMENT:
        f = _tissue_enrichment
    else:
        raise ValueError("Invalid method")

    tissue_expression = tissue_expression.groupby(gene_column).apply(
        lambda x: f(tissue_expression=x, expression_column=expression_column),
        include_groups=True,
    )
    return tissue_expression.reset_index(drop=True)


def pem(
    tissue_expression: pd.DataFrame,
    gene_column: str,
    tissue_column: str,
    expression_column: str,
) -> pd.DataFrame:
    """Calculates the Preferential Expression Measure (PEM) for gene expression data [1]_.

    Args:
        tissue_expression: The DataFrame of containing the gene expression data
        across different tissues.
        gene_column: The column containing gene identifiers.
        tissue_column: The column containing tissue identifiers.
        expression_column: The column containing the gene expression values.

    Returns:
        The input DataFrame with an added column reporting tissue specificity.

    References:
        [1] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5444245/.

        Example:
            >>> import pandas as pd
            >>> from omics_tools.tissue_specificity import pem
            >>> tissue_expression = pd.DataFrame({
            >>>     "gene": ["GFAP"] * 3 + ["INS"] * 3,
            >>>     "tissue": ["brain", "liver", "pancreas"] * 2,
            >>>     "expression": [20, 1, 1, 1, 10, 1],
            >>> })
            >>> result = pem(tissue_expression, gene_column="gene", expression_column="expression", tissue_column="tissue")
            >>> print(result)
    """
    # Pivot the DataFrame to have genes as rows and tissues as columns
    observed = tissue_expression.pivot(
        index=gene_column, columns=tissue_column, values=expression_column
    )

    # Calculate expected values
    tissue_sum = observed.to_numpy().sum(axis=0)
    gene_sum = observed.to_numpy().sum(axis=1, keepdims=True)
    total_sum = tissue_sum.sum()
    expected = gene_sum * (tissue_sum / total_sum)

    # Calculate PEM using log10 of the ratio of observed to expected values
    pem_values = pd.DataFrame(
        np.log10(observed.to_numpy() / (expected + np.finfo(float).eps)),
        columns=observed.columns,
        index=observed.index,
    )

    # Convert the PEM DataFrame back to long format
    pem_long = (
        pem_values.reset_index()
        .melt(
            id_vars=gene_column,
            var_name=tissue_column,
            value_name="pem",
        )
        .sort_values(by=gene_column)
        .reset_index(drop=True)
    )

    return pd.merge(tissue_expression, pem_long, on=[gene_column, tissue_column])
