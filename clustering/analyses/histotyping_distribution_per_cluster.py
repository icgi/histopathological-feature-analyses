import argparse
from pathlib import Path

import polars as pl

import plots


def get_variant(name):
    if "lowres" in name:
        assert "highres" not in name
        variant = "lowres"
    elif "highres" in name:
        assert "lowres" not in name
        variant = "highres"
    else:
        print("Invalid input")
        exit()
    return variant


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

    variant = get_variant(args.input.parent.name)

    if args.output is None:
        output_dir = args.input.parent
    else:
        output_dir = args.output

    df = pl.read_csv(args.input)
    assert "histotyping_score" in df.columns
    assert "cluster" in df.columns
    df = df.select(["cluster", "histotyping_score"])
    print("Input")
    print(df)
    # Order by histotyping score average
    df_summary = pl.read_csv(args.input.with_name(args.input.stem + "_summary.csv"))
    cluster_names = (
        df_summary.sort("percentile-50", descending=True)
        .get_column("cluster")
        .to_list()
    )
    cluster_data = [
        df.filter(pl.col("cluster") == c).get_column("histotyping_score").to_numpy()
        for c in cluster_names
    ]

    output_path = output_dir.joinpath(
        f"per-cluster_ht-score-distribution_{variant}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plots.plot_violins(
        output_path,
        cluster_data,
        color="#f08024",
        xlabel="Cluster #",
        ylabel="Histotyping score",
        xticklabels=cluster_names,
        fig_height=5,
        fig_width=25,
        save_svg=False,
    )


if __name__ == "__main__":
    main()
