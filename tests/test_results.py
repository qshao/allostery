from __future__ import annotations

import csv
from pathlib import Path

from allostery.io import write_pair_scores_csv
from allostery.pipeline.score import PairScore


def _pair_score(
    *,
    residue_i_index: int,
    residue_i_chain: str,
    residue_i_number: int,
    residue_i_name: str,
    residue_j_index: int,
    residue_j_chain: str,
    residue_j_number: int,
    residue_j_name: str,
    score: float,
) -> PairScore:
    return {
        "residue_i": {
            "index": residue_i_index,
            "chain_id": residue_i_chain,
            "residue_number": residue_i_number,
            "name": residue_i_name,
        },
        "residue_j": {
            "index": residue_j_index,
            "chain_id": residue_j_chain,
            "residue_number": residue_j_number,
            "name": residue_j_name,
        },
        "score": score,
    }


def test_write_pair_scores_csv_writes_expected_columns_and_sorted_rows(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "scores" / "ranked_pairs.csv"
    scores = [
        _pair_score(
            residue_i_index=3,
            residue_i_chain="B",
            residue_i_number=11,
            residue_i_name="SER",
            residue_j_index=8,
            residue_j_chain="B",
            residue_j_number=17,
            residue_j_name="THR",
            score=0.25,
        ),
        _pair_score(
            residue_i_index=0,
            residue_i_chain="A",
            residue_i_number=1,
            residue_i_name="GLY",
            residue_j_index=4,
            residue_j_chain="A",
            residue_j_number=5,
            residue_j_name="LEU",
            score=0.95,
        ),
        _pair_score(
            residue_i_index=1,
            residue_i_chain="A",
            residue_i_number=2,
            residue_i_name="ALA",
            residue_j_index=6,
            residue_j_chain="A",
            residue_j_number=9,
            residue_j_name="TYR",
            score=0.6,
        ),
    ]

    write_pair_scores_csv(output_path, scores)

    assert output_path.exists()
    with output_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == [
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
    assert rows == [
        {
            "rank": "1",
            "score": "0.95",
            "residue_i_index": "0",
            "residue_i_chain": "A",
            "residue_i_number": "1",
            "residue_i_name": "GLY",
            "residue_j_index": "4",
            "residue_j_chain": "A",
            "residue_j_number": "5",
            "residue_j_name": "LEU",
        },
        {
            "rank": "2",
            "score": "0.6",
            "residue_i_index": "1",
            "residue_i_chain": "A",
            "residue_i_number": "2",
            "residue_i_name": "ALA",
            "residue_j_index": "6",
            "residue_j_chain": "A",
            "residue_j_number": "9",
            "residue_j_name": "TYR",
        },
        {
            "rank": "3",
            "score": "0.25",
            "residue_i_index": "3",
            "residue_i_chain": "B",
            "residue_i_number": "11",
            "residue_i_name": "SER",
            "residue_j_index": "8",
            "residue_j_chain": "B",
            "residue_j_number": "17",
            "residue_j_name": "THR",
        },
    ]
