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


def write_pair_scores_csv(path: str | Path, scores: Iterable[PairScore]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ranked_scores = sorted(list(scores), key=lambda pair_score: pair_score["score"], reverse=True)
    has_edge_types = any("edge_type_probabilities" in pair_score for pair_score in ranked_scores)
    fieldnames = [*CSV_COLUMNS, "edge_type_probabilities"] if has_edge_types else CSV_COLUMNS
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
            if has_edge_types:
                row["edge_type_probabilities"] = json.dumps(pair_score.get("edge_type_probabilities", []))
            writer.writerow(row)


__all__ = ["CSV_COLUMNS", "write_pair_scores_csv"]
