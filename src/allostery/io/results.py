from __future__ import annotations

import csv
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

    ranked_scores = sorted(scores, key=lambda pair_score: pair_score["score"], reverse=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for rank, pair_score in enumerate(ranked_scores, start=1):
            writer.writerow(
                {
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
            )


__all__ = ["CSV_COLUMNS", "write_pair_scores_csv"]
