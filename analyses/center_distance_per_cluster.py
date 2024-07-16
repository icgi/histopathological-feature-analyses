import argparse
from pathlib import Path

import polars as pl

import plots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        metavar="PATH",
        type=Path,
        help="Input csv with clusters and histotyping score",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        type=Path,
        help="Custom output dir. Input parent if not given",
    )
    args = parser.parse_args()

    if args.output is None:
        # output_dir = args.input.parent.joinpath("per-cluster_ht-score-distribution")
        output_dir = args.input.parent
    else:
        output_dir = args.output

    df = pl.read_csv(args.input)
    assert "histotyping_score" in df.columns
    assert "cluster" in df.columns
    assert "distance" in df.columns
    df = df.select(["cluster", "distance", "histotyping_score"])
    print("Input")
    print(df)
    # Order by histotyping score average
    cluster_names = list(
        df.group_by("cluster").mean().sort("histotyping_score").get_column("cluster")
    )
    cluster_data = [
        df.filter(pl.col("cluster") == c).get_column("distance").to_numpy()
        for c in cluster_names
    ]

    output_path = output_dir.joinpath("per-cluster_distance-distribution.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plots.plot_violins(
        output_path,
        cluster_data,
        color="#123456",
        xlabel="Cluster #",
        ylabel="Euclidean distance",
        xticklabels=cluster_names,
        y_max=45,
        fig_height=5,
        fig_width=25,
        save_svg=False,
    )


if __name__ == "__main__":
    main()
