import numpy as np
import pandas as pd
from fisher import pvalue_npy
from statsmodels.stats.multitest import multipletests


def fisher(
    query: set[str], annotation_mapper: dict[str, set[str]], space_size: int
) -> pd.DataFrame:
    """Performs functional enrichment analysis using Fisher's exact test.

    Args:
        query: A set of query items.
        annotation_mapper: A dictionary where keys are annotation IDs
            and values are sets of items annotated with that ID.
        space_size: The size of the space from which the query is drawn.

    Returns:
        A DataFrame containing the results of the enrichment analysis, with columns:
            - query: The original query set.
            - annotation_id: The annotation IDs.
            - annotation: The annotation sets.
            - overlap: The overlap between the query and the annotation sets.
            - query_length: The length of the query set.
            - annotation_length: The length of the annotation sets.
            - overlap_length: The length of the overlap sets.
            - pvalue: The p-values of the overlaps.
            - pfdr:  The adjusted p-values using the FDR method (Benjamini/Hochberg).

    Example:
        >>> from omics_tools.enrichment import fisher
        >>> query = {"gene1", "gene2", "gene3"}
        >>> annotation_mapper = {
        >>>     "pathway1": {"gene1", "gene4", "gene5"},
        >>>     "pathway2": {"gene2", "gene6"},
        >>>     "pathway3": {"gene7", "gene8"}
        >>> }
        >>> space_size = 1000
        >>> results = fisher(query, annotation_mapper, space_size)
        >>> print(results)
    """
    enrichment_results = pd.DataFrame(
        columns=[
            "query",
            "annotation_id",
            "annotation",
            "overlap",
            "query_length",
            "annotation_length",
            "overlap_length",
            "pvalue",
            "pfdr",
        ]
    )
    enrichment_results["annotation_id"] = list(annotation_mapper.keys())
    enrichment_results["annotation"] = list(annotation_mapper.values())
    enrichment_results["query"] = [query] * len(enrichment_results)
    enrichment_results["overlap"] = enrichment_results.apply(
        lambda x: x["annotation"] & x["query"], axis=1
    )
    enrichment_results = enrichment_results.assign(
        **{
            f"{column}_length": lambda x, col=column: x[col].apply(len)
            for column in ["query", "annotation", "overlap"]
        }
    )

    cm = pd.DataFrame(
        {
            "c1": enrichment_results["overlap_length"],
            "c2": enrichment_results["annotation_length"]
            - enrichment_results["overlap_length"],
            "c3": enrichment_results["query_length"]
            - enrichment_results["overlap_length"],
        }
    )
    cm["c4"] = space_size - cm.sum(axis=1).to_numpy()
    cm = cm.to_numpy(dtype=np.uint)

    pvalue_left_tail, pvalue_right_tail, pvalue_two_tail = pvalue_npy(
        cm[:, 0], cm[:, 1], cm[:, 2], cm[:, 3]
    )
    # odds = (cm[:, 0] * cm[:, 3]) / (cm[:, 1] * cm[:, 2])

    enrichment_results["pvalue"] = pvalue_right_tail
    enrichment_results["pfdr"] = multipletests(
        enrichment_results["pvalue"], method="fdr_bh"
    )[1]

    return enrichment_results.drop(columns="annotation")
