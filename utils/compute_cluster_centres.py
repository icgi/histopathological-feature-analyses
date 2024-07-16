import argparse
from pathlib import Path
import time

import polars as pl


def check_tile_paths(df_class, df_feat):
    """Check that all tiles with features have a cluster class"""
    paths_class = df_class.get_column("path").unique().to_list()
    paths_feat = df_feat.get_column("path").unique().to_list()
    assert set(paths_feat).difference(paths_class) == set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tile_features",
        metavar="PATH",
        type=Path,
        help=".csv with tile features",
    )
    parser.add_argument(
        "classification",
        metavar="PATH",
        type=Path,
        help=".csv with tile cluster classifications",
    )
    parser.add_argument(
        "output",
        metavar="PATH",
        type=Path,
        help=".csv with resulting cluster centres",
    )
    args = parser.parse_args()

    assert args.tile_features.is_file()
    assert args.tile_features.suffix == ".csv"
    assert args.classification.is_file()
    assert args.classification.suffix == ".csv"
    assert args.output.suffix == ".csv"

    print("Read classification file")
    start = time.time()
    df_class = (
        pl.scan_csv(args.classification)
        .select(["path", "cluster"])
        .collect(streaming=True)
    )
    print(df_class)
    print(f"Elapsed: {time.time() - start:.0f} s")
    print("Read tile feature file")
    start = time.time()
    df_feat = pl.read_csv(args.tile_features)
    print(df_feat)
    print(f"Elapsed: {time.time() - start:.0f} s")

    check_tile_paths(df_class, df_feat)

    print("Compute average tile features per cluster")
    start = time.time()
    df_result = (
        df_feat.join(df_class, on="path", how="left", coalesce=True)
        .drop("path")
        .group_by("cluster")
        .mean()
        .sort("cluster")
    )
    print(f"Elapsed: {time.time() - start:.0f} s")
    print(df_result)

    print(f"Write '{args.output}'")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_result.write_csv(args.output)


if __name__ == "__main__":
    main()
