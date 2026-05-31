from __future__ import annotations

import argparse
from pathlib import Path

from allostery.pipeline.score import score_trajectory
from allostery.pipeline.train import train_model


def _format_residue(residue: dict[str, object]) -> str:
    return f"{residue['chain_id']}:{residue['residue_number']} {residue['name']}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a small relational model and print top residue pairs.")
    parser.add_argument("pdb_path", type=Path, help="Path to a multi-model PDB trajectory")
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--horizon-size", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--consistency-weight", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    result = train_model(
        pdb_path=args.pdb_path,
        window_size=args.window_size,
        horizon_size=args.horizon_size,
        stride=args.stride,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        consistency_weight=args.consistency_weight,
    )
    ranked_pairs = score_trajectory(
        model=result.model,
        pdb_path=args.pdb_path,
        window_size=args.window_size,
        horizon_size=args.horizon_size,
        stride=args.stride,
    )

    top_k = min(args.top_k, len(ranked_pairs))
    print(f"trained on {result.num_samples} windows, last_loss={result.last_loss:.6f}")
    print(f"top {top_k} residue pairs")
    print("rank  score      residue_i        residue_j")
    for rank, entry in enumerate(ranked_pairs[:top_k], start=1):
        print(
            f"{rank:>4}  {entry['score']:<9.6f}  "
            f"{_format_residue(entry['residue_i']):<14}  {_format_residue(entry['residue_j'])}"
        )


if __name__ == "__main__":
    main()
