from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from allostery.pipeline.score import PairScore

CSV_COLUMNS = [
    "rank",
    "score",
    "residue_i_index",
    "residue_i_chain",
    "residue_i_number",
    "residue_i_name",
    "residue_j_index",
    "residue_j_chain",
    "residue_j_number",
    "residue_j_name",
]
OPTIONAL_COLUMNS = [
    "support_count",
    "mean_distance",
    "edge_type_probabilities",
    "edge_type_stddev",
    "influence_i_on_j",
    "influence_j_on_i",
]


def write_pair_scores_csv(path: str | Path, scores: Iterable[PairScore]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ranked_scores = sorted(list(scores), key=lambda pair_score: pair_score["score"], reverse=True)
    present_optional = [
        col for col in OPTIONAL_COLUMNS
        if any(col in pair_score for pair_score in ranked_scores)
    ]
    fieldnames = [*CSV_COLUMNS, *present_optional] if present_optional else CSV_COLUMNS
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, pair_score in enumerate(ranked_scores, start=1):
            row = {
                "rank": rank,
                "score": pair_score["score"],
                "residue_i_index": pair_score["residue_i"]["index"],
                "residue_i_chain": pair_score["residue_i"]["chain_id"],
                "residue_i_number": pair_score["residue_i"]["residue_number"],
                "residue_i_name": pair_score["residue_i"]["name"],
                "residue_j_index": pair_score["residue_j"]["index"],
                "residue_j_chain": pair_score["residue_j"]["chain_id"],
                "residue_j_number": pair_score["residue_j"]["residue_number"],
                "residue_j_name": pair_score["residue_j"]["name"],
            }
            if present_optional:
                if "support_count" in pair_score:
                    row["support_count"] = pair_score["support_count"]
                if "mean_distance" in pair_score:
                    row["mean_distance"] = pair_score["mean_distance"]
                if "edge_type_probabilities" in pair_score:
                    row["edge_type_probabilities"] = json.dumps(pair_score["edge_type_probabilities"])
                if "edge_type_stddev" in pair_score:
                    row["edge_type_stddev"] = json.dumps(pair_score["edge_type_stddev"])
                if "influence_i_on_j" in pair_score:
                    row["influence_i_on_j"] = pair_score["influence_i_on_j"]
                if "influence_j_on_i" in pair_score:
                    row["influence_j_on_i"] = pair_score["influence_j_on_i"]
            writer.writerow(row)


__all__ = ["CSV_COLUMNS", "write_pair_scores_csv"]
