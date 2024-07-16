import argparse
from pathlib import Path
import shutil

import numpy as np
import polars as pl
from tqdm import tqdm


def filter_on_histotyping(df, percentile_low, percentile_high, size=None):
    threshold_low = np.percentile(df.get_column("histotyping_score"), percentile_low)
    threshold_high = np.percentile(df.get_column("histotyping_score"), percentile_high)
    df_f = df.filter(
        (pl.col("histotyping_score") > threshold_low)
        & (pl.col("histotyping_score") < threshold_high)
    )
    size = 1 if size is None else size
    if len(df_f) < size:
        print(
            f"WARNING: skip filtering since result {len(df_f)} < selection size {size}"
        )
        return df
    else:
        return df_f


def filter_max_per_scan(df, max_size, selection):
    scans = sorted(df.get_column("scan").unique().to_list())
    assert len(scans) > 0, df
    dfs = []
    for scan in scans:
        df_s = df.filter(pl.col("scan") == scan)
        size = min(max_size, len(df_s))
        if selection == "random":
            tile_path_selection = np.random.choice(
                df_s.get_column("path"),
                size,
                replace=False,
            )
        elif selection == "distance":
            df_s = df_s.sort("distance")
            tile_path_selection = df_s.get_column("path")[:size]
        else:
            print("Invalid selection. This should be unreachable")
            exit()
        dfs.append(df_s.filter(pl.col("path").is_in(tile_path_selection)))
    return pl.concat(dfs)


def add_scan(df, filename):
    if "_uni_" in filename:
        df = df.with_columns(
            pl.col("path")
            .map_elements(
                lambda p: Path(p).parents[2].name,
                return_dtype=str,
            )
            .alias("scan")
        )
    else:
        df = df.with_columns(
            pl.col("path")
            .map_elements(
                lambda p: Path(p).parents[0].name + "_" + Path(p).parents[1].name,
                return_dtype=str,
            )
            .alias("scan")
        )
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        metavar="PATH",
        type=Path,
        help="Input cluster csv",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        type=Path,
        help="Custom output folder. Otherwise input parent",
    )
    parser.add_argument(
        "--selection",
        metavar="STR",
        type=str,
        default="distance",
        choices=["distance", "random"],
        help="How to select tiles per cluster to copy [distance]",
    )
    parser.add_argument(
        "--size",
        metavar="INT",
        type=int,
        default=10,
        help="How many tiles per cluster to copy [10]",
    )
    parser.add_argument(
        "--max_per_scan",
        metavar="INT",
        type=int,
        default=10,
        help="How many tiles per scan to copy [10]",
    )
    parser.add_argument(
        "--exclude",
        action="store_true",
        help="For each cluster, exclude extreme tiles based on histotyping score",
    )
    args = parser.parse_args()

    if args.output is None:
        name = f"per-cluster-{args.size}_per-scan-{args.max_per_scan}"
        output_root = args.input.parent.joinpath(f"example-tiles_{name}")
    else:
        output_root = args.output

    df = pl.read_csv(args.input)
    assert "histotyping_score" in df.columns
    if "scan" not in df.columns:
        df = add_scan(df, args.input.stem)
    print("Input")
    print(df)
    print("Tiles per cluster")
    print(df.group_by("cluster").len().sort("cluster"))

    clusters = sorted(df.get_column("cluster").unique().to_list())

    dfs_selected = []
    for cluster in tqdm(clusters):
        output_dir = output_root.joinpath(f"cluster-{cluster:02d}")
        output_dir.mkdir(parents=True, exist_ok=True)
        df_c = df.filter(pl.col("cluster") == cluster)
        if args.exclude and "histotyping_score" in df_c.columns:
            df_c = filter_on_histotyping(df_c, 5, 95, args.size)
        if args.max_per_scan is not None:
            df_c = filter_max_per_scan(df_c, args.max_per_scan, args.selection)

        size = min(args.size, len(df_c))
        if args.selection == "random":
            tile_path_selection = np.random.choice(
                df_c.get_column("path"),
                size,
                replace=False,
            )
        elif args.selection == "distance":
            df_c = df_c.sort("distance")
            tile_path_selection = df_c.get_column("path")[:size]
        else:
            print("Invalid selection. This should be unreachable")
            exit()
        dfs_selected.append(df_c.filter(pl.col("path").is_in(tile_path_selection)))
        for tile_path in tile_path_selection:
            tile_path = Path(tile_path)
            scan = tile_path.parents[2].name
            output_path = output_dir.joinpath(f"{scan}___{tile_path.name}")
            if tile_path.exists():
                shutil.copy(tile_path, output_path)

    df = pl.concat(dfs_selected).sort(by=["cluster", "distance"])
    output_path = output_root.joinpath("selection_info.csv")
    print(f"Write selection info to '{output_path}'")
    df.write_csv(output_path)


if __name__ == "__main__":
    main()
