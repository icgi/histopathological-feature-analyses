import argparse
from pathlib import Path

import numpy as np
import scipy
import polars as pl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        metavar="PATH",
        type=Path,
        help="Input cluster csv",
    )
    parser.add_argument(
        "scores",
        metavar="PATH",
        type=Path,
        help="Input csv with all histotyping scores",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        type=Path,
        help="Optional custom output folder. Default is input folder",
    )
    parser.add_argument(
        "--selection",
        metavar="STR",
        type=str,
        default="distance",
        choices=["distance", "random"],
        help="How to select tiles per cluster [distance] if --size is given",
    )
    parser.add_argument(
        "--size",
        metavar="INT",
        type=int,
        help="How many tiles per cluster to include. All if not given",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    assert args.input.exists()
    assert args.input.suffix == ".csv"

    output_dir = args.input.parent if args.output is None else args.output
    output_path = output_dir.joinpath(f"ht-score_{args.input.name}")
    if output_path.exists():
        print("Output exists. Terminating")
        exit()

    df = pl.read_csv(args.input)
    print("Input")
    print(df)
    df = df.with_columns(
        pl.col("path")
        .map_elements(lambda p: Path(p).parents[2].name, return_dtype=str)
        .alias("scan")
    )
    df = df.with_columns(
        pl.col("path")
        .map_elements(lambda p: Path(p).stem.split("___")[-1], return_dtype=str)
        .alias("coordinate")
    )
    df = df.select(["path", "scan", "coordinate", "cluster", "distance"])
    print("After adding columns")
    print(df)
    print("Tiles per cluster")
    print(df.group_by("cluster").len().sort("cluster"))

    clusters = sorted(df.get_column("cluster").unique().to_list())

    if args.size is None:
        df_inc = df
    else:
        dfs = []
        for cluster in clusters:
            df_c = df.filter(pl.col("cluster") == cluster)
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
            dfs.append(df_c.filter(pl.col("path").is_in(tile_path_selection)))
        df_inc = pl.concat(dfs)

    print("Data to analyse")
    # df_inc = df_inc.drop("path")
    print(df_inc)
    print("Append histotyping scores")
    df = df_inc.join(
        pl.read_csv(args.scores), how="left", on=["scan", "coordinate"], coalesce=True
    )
    if "histotyping_score" not in df.columns:
        print("No histotyping score")
        exit()
    print(df)

    records = []
    for cluster in clusters:
        scores = df.filter(pl.col("cluster") == cluster).get_column("histotyping_score")
        summary = scipy.stats.describe(scores)
        records.append(
            {
                "cluster": cluster,
                "size": summary.nobs,
                "mean": summary.mean,
                "variance": summary.variance,
                "stddev": np.sqrt(summary.variance),
                "percentile-05": np.percentile(scores, 5),
                "percentile-25": np.percentile(scores, 25),
                "percentile-50": np.percentile(scores, 50),
                "percentile-75": np.percentile(scores, 75),
                "percentile-95": np.percentile(scores, 95),
                "min": summary.minmax[0],
                "max": summary.minmax[1],
            }
        )
    df_summary = pl.DataFrame(records).sort("mean")
    pl.Config.set_tbl_rows(len(clusters))
    pl.Config.set_tbl_cols(10)
    print(df_summary)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing '{output_path}'")
    df.write_csv(output_path)
    df_summary.write_csv(output_path.with_name(f"{output_path.stem}_summary.csv"))


if __name__ == "__main__":
    main()
