import pandas as pd

from omics_tools import enrichment


def test_fisher():
    query = {"A", "B", "C"}
    annotation_mapper = {
        "pathway_1": {"A", "B", "C", "D", "E"},
        "pathway_2": {"A", "B"},
    }
    space_size = 100
    observed = enrichment.fisher(
        query=query, annotation_mapper=annotation_mapper, space_size=space_size
    )
    observed[["pvalue", "pfdr"]] = observed[["pvalue", "pfdr"]].round(5)
    expected = pd.DataFrame(
        {
            "query": [query] * len(annotation_mapper),
            "annotation_id": annotation_mapper.keys(),
            "overlap": [{"A", "B", "C"}, {"A", "B"}],
            "query_length": [3, 3],
            "annotation_length": [5, 2],
            "overlap_length": [3, 2],
            "pvalue": [0.00006, 0.00061],
            "pfdr": [0.00012, 0.00061],
        }
    )
    pd.testing.assert_frame_equal(observed, expected)
