#!/usr/bin/env python3
"""Random clustering module for Omnibenchmark."""
from __future__ import annotations
import argparse
import gzip
from pathlib import Path
import sys
import numpy as np
import pandas as pd

def generate_k_grid(base_k: int) -> list[int]:
    """Return the [k-2, k-1, k, k+1, k+2] grid with each value >= 2."""
    return [max(2, base_k + i) for i in range(-2, 3)]

def assign_random_clusters(num_rows: int, ks: list[int], seed: int | None = None) -> np.ndarray:
    """Return matrix of random cluster assignments per k in ks."""
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.integers(1, k + 1, size=num_rows) for k in ks])

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assign random cluster labels.")
    parser.add_argument("--data.true_labels", dest="data_true_labels", type=Path)
    parser.add_argument("--data.matrix", dest="data_matrix", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--num-clusters", dest="num_clusters", required=True, type=int)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)

def main(argv: list[str]) -> int:
    args = parse_args(argv)
    sep = "\t" if any(s in args.data_matrix.suffixes for s in [".tsv", ".txt"]) else ","
    df = pd.read_csv(args.data_matrix, sep=sep, header=None, compression="gzip")
    base_k = args.num_clusters
    if args.data_true_labels:
        labels_df = pd.read_csv(args.data_true_labels, header=None, compression="gzip")
        base_k = int(labels_df.iloc[:, 0].max())
    ks = generate_k_grid(base_k)
    label_matrix = assign_random_clusters(len(df), ks, args.seed)
    header = np.array([[f"k={k}" for k in ks]])
    output_matrix = np.vstack([header, label_matrix.astype(str)])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output_dir / f"{args.name}_ks_range.labels.gz", "wt") as f:
        np.savetxt(f, output_matrix, fmt="%s", delimiter=",")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
