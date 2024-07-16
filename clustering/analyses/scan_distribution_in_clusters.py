import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import plots

dpi = 200


def percentage_distribution(df):
    """
    histogram with bins:
    [0], [1, 4], [5, 9], ..., [96, 100]

    np.histogram has half-open bin ranges:
        [low, high), except for the last which is [low, high]
    """
    num_total = len(df)
    df_zero = df.filter(pl.col("percentage") == 0)
    num_zero = len(df_zero)
    df_pos = df.filter(pl.col("percentage") > 0)
    num_pos = len(df_pos)
    info = {
        "total_count": num_total,
        "zero_count": num_zero,
        "zero_percent": num_zero / num_total * 100,
        "positive_count": num_pos,
        "positive_percent": num_pos / num_total * 100,
    }
    str_ex = f"{num_zero:>4} / {num_total:>4} = {info['zero_percent']:>5.2f}%"
    str_in = f"{num_pos:>4} / {num_total:>4} = {info['positive_percent']:>5.2f}%"
    print(f"Scans not represented: {str_ex}")
    print(f"Scans represented:     {str_in}")
    assert num_zero + num_pos == num_total
    percentages = df_pos.get_column("percentage")
    bins = np.arange(0, 101, 5)
    hist, _ = np.histogram(percentages, bins=bins, range=(0, 100))

    info["[0]"] = num_zero
    bin_start = 0
    for count, bin_end in zip(hist, bins[1:]):
        if bin_start == 0:
            start_symbol = "("
        else:
            start_symbol = "["
        if bin_end == 100:
            end_symbol = "]"
        else:
            end_symbol = ")"
        key = f"{start_symbol}{bin_start}-{bin_end}{end_symbol}"
        info[key] = count
        bin_start = bin_end

    return info, str_in, str_ex, percentages.to_numpy()


def plot_cluster_histogram(df, output_path, title):
    x = df.get_column("histotyping_score")
    y = df.get_column("percentage")

    fig, ax = plt.subplots()
    fig.set_size_inches(20, 5)
    ax.bar(x, y, width=0.001)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 1)
    ax.set_ylabel("Percentage of scan")
    ax.set_xlabel("Scan histotyping score")
    ax.set_title(title)
    plt.savefig(output_path, dpi=100, bbox_inches="tight", transparent=False)
    plt.close()


def get_input(path):
    assert path.is_file()
    assert path.suffix == ".csv"
    q = pl.scan_csv(path).select(["scan", "cluster", "histotyping_score"])
    return q.collect(streaming=True)


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


def from_scratch(input_path, output_root):
    variant = get_variant(input_path.parent.name)
    df = get_input(input_path)
    df_summary = pl.read_csv(
        input_path.with_name(input_path.stem + "_summary.csv")
    ).sort("percentile-50")
    df_score = (
        df.select(["scan", "histotyping_score"]).group_by("scan").mean().sort("scan")
    )
    df_count = df.group_by("scan").len().sort("scan").rename({"len": "total"})

    print(df)
    print(df_summary)

    output_root = input_path.parent if output_root is None else output_root
    output_root = output_root.joinpath(f"scan-cluster-distribution/{variant}")
    output_dir = output_root.joinpath("per-cluster")
    output_dir.mkdir(parents=True, exist_ok=True)
    info_records = []
    cluster_percentages = []
    for row in df_summary.iter_rows(named=True):
        cluster = row["cluster"]
        print(f"Cluster {cluster}")
        df_c = df.filter(pl.col("cluster") == cluster)
        df_count_c = df_c.group_by("scan").len().sort("scan")
        df_count_c = df_count_c.join(
            df_count, on="scan", how="full", coalesce=True
        ).fill_null(0)
        df_count_c = df_count_c.with_columns(
            (pl.col("len") / pl.col("total") * 100).alias("percentage")
        )
        df_count_c = df_count_c.join(df_score, on="scan").sort("histotyping_score")
        # print(df_count_c)
        info = {
            "cluster": cluster,
            "ht-median": row["percentile-50"],
            "ht-mean": row["mean"],
        }
        info_counts, str_in, str_ex, percentages = percentage_distribution(df_count_c)
        info.update(info_counts)
        info_records.append(info)
        cluster_percentages.append(percentages)
        output_path = output_dir.joinpath(
            f"scan-percentage_{variant}_cluster-{cluster:02d}.png"
        )
        title = f"Cluster {cluster}\n"
        title += f"Scans not represented: {str_ex}\n"
        title += f"Scans represented:     {str_in}"
        plot_cluster_histogram(df_count_c, output_path, title)

    plots.plot_boxes(
        output_root.joinpath(f"cluster-distribution-in-scan-boxplot-log_{variant}.png"),
        cluster_percentages,
        # color=colours_rgb,
        color="#1f77b4",
        # xlabel="Cluster #",
        # ylabel="Histotyping score",
        # xticklabels=cluster_names,
        fig_height=5,
        fig_width=25,
        save_svg=False,
    )

    output_path = output_root.joinpath(f"count-distribution_{variant}.csv")
    print(f"Write summary to '{output_path}'")
    df = pl.DataFrame(info_records)
    print(df)
    df.write_csv(output_path)
    return output_path


def plot_presence(df, output_path, save_svg):
    clusters = df.get_column("cluster").to_list()
    values = df.get_column("positive_percent").to_list()
    # values = np.random.randint(0, 100, 100)

    fig, ax = plt.subplots(figsize=(5, 25), layout="constrained")
    rects = ax.barh(
        range(len(clusters)), values, align="center", height=0.8, color="#4878b6"
    )

    large_values = [f"{x:.0f}" if x > 95 else "" for x in values]
    small_values = [f"{x:.0f}" if x <= 95 else "" for x in values]
    ax.bar_label(rects, small_values, padding=5, color="black")
    ax.bar_label(rects, large_values, padding=-20, color="white")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.set_xlim(0, 100)
    ax.set_ylim(-1, 100)
    ax.set_xlabel("Percentage of scans containing cluster")
    ax.set_ylabel("Cluster")
    ax.set_yticks(ticks=range(len(clusters)), labels=clusters)
    plt.savefig(output_path, dpi=100, bbox_inches="tight", transparent=False)
    if save_svg:
        plt.savefig(
            output_path.with_suffix(".svg"),
            format="svg",
            dpi=dpi,
            bbox_inches="tight",
            transparent=True,
        )


def write_colorbar(output_path, cmap, axis=False, save_svg=False):
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient)).T
    gradient = np.flipud(gradient)
    # Create figure and adjust figure height to number of colormaps
    fig, ax = plt.subplots()
    fig.set_size_inches(1, 10)
    ax.imshow(gradient, aspect="auto", cmap=mpl.colormaps[cmap])
    if axis:
        ytick_locations = [k * 255 / 4 for k in reversed(range(5))]
        ytick_labels = [f"{t / 255:.2f}" for t in ytick_locations]
        ytick_labels[0] = "0"
        ytick_labels[-1] = "1"
        ax.set_yticks(ticks=ytick_locations, labels=ytick_labels)
        ax.tick_params(
            axis="x",
            which="both",
            bottom=False,
            top=False,
            labelbottom=False,
        )
    else:
        ax.axis("off")
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", transparent=True)
    if save_svg:
        plt.savefig(
            output_path.with_suffix(".svg"),
            format="svg",
            dpi=dpi,
            bbox_inches="tight",
            transparent=True,
        )


def plot_distribution(df, output_path, variant, save_svg):
    hist_cols = [
        "(0-5)",
        "[5-10)",
        "[10-15)",
        "[15-20)",
        "[20-25)",
        "[25-30)",
        "[30-35)",
        "[35-40)",
        "[40-45)",
        "[45-50)",
        "[50-55)",
        "[55-60)",
        "[60-65)",
        "[65-70)",
        "[70-75)",
        "[75-80)",
        "[80-85)",
        "[85-90)",
        "[90-95)",
        "[95-100]",
    ]
    cmap = "viridis"
    x_labels = [int(x.strip("[]()").split("-")[-1]) for x in hist_cols]
    clusters = df.get_column("cluster").to_list()
    # num_pos = df.get_column("positive_count").to_numpy()
    # values = df.select(hist_cols).to_numpy() / num_pos.reshape((100, 1))
    values = df.select(hist_cols).to_numpy() / 1082
    alpha = np.zeros_like(values).astype(float)
    alpha[values > 0] = 1.0

    write_colorbar(output_path.with_name("colorbar.png"), cmap, False, True)

    fig, ax = plt.subplots(figsize=(5, 25), layout="constrained")

    if variant == "lowres":
        vmax = 0.5
    elif variant == "highres":
        vmax = 0.9
    else:
        print("Invalid variant")
        exit()
    ax.imshow(
        values,
        cmap=mpl.colormaps[cmap],
        aspect="auto",
        extent=[0, 20, -0.5, 99.5],
        origin="lower",
        vmin=0,
        alpha=alpha,
        vmax=vmax,
        # norm="log"
    )

    ax.set_xlim(0, len(x_labels))
    ax.set_ylim(-1, 100)
    ax.set_xlabel("Percentage of scan containing cluster")
    ax.set_ylabel("Cluster")
    ax.set_xticks(ticks=range(1, len(x_labels) + 1), labels=x_labels)
    ax.set_yticks(ticks=range(len(clusters)), labels=clusters)
    print("VMAX:", vmax)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", transparent=False)
    if save_svg:
        plt.savefig(
            output_path.with_suffix(".svg"),
            format="svg",
            dpi=dpi,
            bbox_inches="tight",
            transparent=True,
        )


def from_result(input_path, output_dir=None):
    df = pl.read_csv(input_path)
    variant = input_path.stem.split("_")[-1]
    print(f"Variant {variant}")
    # The numbers in hist_cols are percentages
    output_dir = input_path.parent if output_dir is None else output_dir
    plot_presence(
        df,
        output_dir.joinpath(f"cluster-presence-in-scan_{variant}.png"),
        save_svg=True,
    )
    plot_distribution(
        df,
        output_dir.joinpath(f"cluster-distribution-in-scan_{variant}.png"),
        variant,
        save_svg=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_raw",
        metavar="PATH",
        type=Path,
        help="Csv with clusters and histotyping score. All analyses from scratch",
    )
    parser.add_argument(
        "--input_result",
        metavar="PATH",
        type=Path,
        help="Csv with result from scratch",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        type=Path,
        help="Custom output dir. Input parent if not given",
    )
    args = parser.parse_args()

    if args.input_raw is not None:
        result_path = from_scratch(args.input_raw, args.output)
        from_result(result_path)
    elif args.input_result is not None:
        from_result(args.input_result, args.output)
    else:
        print("No input. Terminating")


if __name__ == "__main__":
    main()
