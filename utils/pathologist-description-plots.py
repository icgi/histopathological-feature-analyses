import argparse
from pathlib import Path
from pprint import pprint
from textwrap import wrap

import cv2
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


class Paths:
    """
    Assumed input: /path/to/<variant>, where <variant> in {lowres,highres} with content

    cluster-stats.csv  # For per-cluster statistics such as ht mean and std dev
    descriptions.csv   # Pathologist descriptions per cluster
    downscaled-tiles-jpg/  # Example tiles per cluster
    ht-score_tile-clusters_pathologist-selection_<variant>.csv  # Info per example tile
    selected-clusters_<variant>.txt  # Included clusters
    """

    def __init__(self, root, variant, output_dir=None):
        self.cluster_stats = root.joinpath("cluster-stats.csv")
        self.descriptions = root.joinpath("descriptions.csv")
        self.downscaled_tiles = root.joinpath("downscaled-tiles-jpg")
        self.tile_info = root.joinpath(
            f"ht-score_tile-clusters_pathologist-selection_{variant}.csv"
        )
        self.included_clusters = root.joinpath(f"selected-clusters_{variant}.txt")
        self.output_dir = (
            output_dir if output_dir is not None else root.joinpath("result-figures")
        )

        assert self.cluster_stats.is_file()
        assert self.descriptions.is_file()
        assert self.downscaled_tiles.is_dir()
        assert self.tile_info.is_file()
        assert self.included_clusters.is_file()


def select_tiles(tile_dir, df_tile, num_tiles):
    suffix = ".jpg"
    df_tile = df_tile.sort("distance")
    cluster_tiles = []
    for tile_name in df_tile.get_column("tile").to_list():
        tile_path = tile_dir.joinpath(f"{tile_name}{suffix}")
        assert tile_path.exists(), tile_path
        cluster_tiles.append(tile_path)
    return cluster_tiles[:num_tiles]


class FigureElement:

    def __init__(self, cluster, cluster_stats, df_tile, tile_dir, description):
        self.num_tiles = 5
        self.cluster = cluster
        self.stats = cluster_stats
        self.stats["std"] = np.sqrt(self.stats["variance"])
        self.description = description
        assert self.cluster == self.stats["cluster"]
        assert self.cluster == int(tile_dir.name.split("-")[-1])

        self.tile_paths = select_tiles(tile_dir, df_tile, self.num_tiles)
        assert len(self.tile_paths) == self.num_tiles


def remove_ticklabels(ax):
    ax.tick_params(
        axis="x",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
    )
    ax.tick_params(
        axis="y",
        which="both",
        left=False,
        right=False,
        labelleft=False,
    )
    return ax


def create_figure(output_dir, figure_idx, elements, variant, debug=False):
    """
    cluster info | tile_00 | ... | tile_0n
                 | pathologist description
    --------------------------------------
    cluster info | tile_10 | ... | tile_1n
                 | pathologist description
    --------------------------------------
                 ...
    --------------------------------------
    cluster info | tile_m0 | ... | tile_mn
                 | pathologist description
    """
    print(f"Figure {figure_idx}")
    save_svg = False
    a4_factor = 0.6
    dpi = 200
    add_panel = False
    num_im_rows = len(elements)
    num_im_cols = len(elements[0].tile_paths)
    num_rows = num_im_rows * 2  # alternating image row and description
    num_cols = num_im_cols + 1  # for cluster info + image row
    fig = plt.figure(figsize=(21 * a4_factor, 29.7 * a4_factor))
    spec = fig.add_gridspec(
        ncols=num_cols,
        nrows=num_rows,
        hspace=0.05,
        wspace=0.05,
        width_ratios=[0.7] + [1] * num_im_cols,
        height_ratios=[3, 1] * num_im_rows,
    )
    for i, element in enumerate(elements):
        if debug:
            pprint(vars(element))

        # Cluster info and stats
        ax_i = fig.add_subplot(spec[2 * i : 2 * i + 2, 0])
        text = (
            rf"$\bf{{Cluster}}$ $\bf{{{element.cluster}}}$"
            "\n"
            f"Median: {element.stats['percentile-50']:.2f}\n"
            f"IQR: ({element.stats['percentile-25']:.2f}—"
            f"{element.stats['percentile-75']:.2f})"
        )
        if debug:
            ax_i = remove_ticklabels(ax_i)
        else:
            if add_panel:
                panel = ["a", "b", "c", "d", "e", "f"][i]
                ax_i.text(
                    0,
                    1,
                    panel,
                    horizontalalignment="left",
                    verticalalignment="top",
                    fontsize=20,
                    fontweight="bold",
                )
            ax_i.text(
                0,
                0.7 if add_panel else 1,
                text,
                horizontalalignment="left",
                verticalalignment="top",
            )
            ax_i.axis("off")

        # Tile row
        for j, tile_path in enumerate(element.tile_paths):
            ax_im = fig.add_subplot(spec[2 * i, j + 1])
            im = cv2.imread(str(tile_path), cv2.IMREAD_COLOR)
            if debug:
                ax_im = remove_ticklabels(ax_im)
            else:
                ax_im.axis("off")
                ax_im.imshow(im[:, :, ::-1])

        # Description
        ax_d = fig.add_subplot(spec[2 * i + 1, 1:])
        if debug:
            ax_d = remove_ticklabels(ax_d)
        else:
            ax_d.text(
                0,
                1,
                "\n".join(wrap(element.description, 110)),
                wrap=True,
                horizontalalignment="left",
                verticalalignment="top",
            )
            ax_d.axis("off")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir.joinpath(f"description-figure_{variant}_{figure_idx}.jpg")
    # TODO: Save eps or svg
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", transparent=False)
    if save_svg:
        plt.savefig(
            output_path.with_suffix(".svg"),
            format="svg",
            dpi=dpi,
            bbox_inches="tight",
            transparent=True,
        )
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        metavar="PATH",
        type=Path,
        help="Path to input folder",
    )
    parser.add_argument(
        "--include",
        metavar="INT",
        type=int,
        nargs="+",
        help="Only include these clusters",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        type=Path,
        help="Custom output dir",
    )
    args = parser.parse_args()

    num_clusters = len(args.include) if args.include is not None else 30
    num_figures = 5
    assert num_clusters % num_figures == 0

    input_root = Path(args.input)
    assert input_root.is_dir()
    variant = input_root.name
    assert variant in {"lowres", "highres"}
    print(f"Input root: {input_root}")

    paths = Paths(input_root, variant, args.output)

    cluster_stats = {}
    df_cluster = pl.read_csv(paths.cluster_stats).sort("percentile-50")
    for row in df_cluster.iter_rows(named=True):
        cluster_stats[row["cluster"]] = row

    with paths.included_clusters.open() as f:
        included_clusters = [int(r.strip().split(" ")[-1]) for r in f.readlines()]

    if args.include is not None:
        included_clusters = [c for c in included_clusters if c in args.include]

    descriptions = {}
    for row in pl.read_csv(paths.descriptions).iter_rows(named=True):
        cluster = row["cluster"]
        if cluster not in included_clusters:
            continue
        descriptions[cluster] = row["description"]
    assert set(included_clusters) == set(descriptions.keys())
    assert len(included_clusters) == num_clusters

    df_tile = pl.read_csv(paths.tile_info)
    df_tile = df_tile.with_columns(
        pl.struct(pl.all())
        .map_elements(
            lambda r: r["scan"] + "___" + Path(r["path"]).stem, return_dtype=str
        )
        .alias("tile")
    )

    # Iter cluster_stats instead of included_clusters since cluster_stats is sorted by
    # histotyping score
    figure_elements = []
    for cluster, stats in cluster_stats.items():
        if cluster not in included_clusters:
            continue
        df_tile_cluster = df_tile.filter(pl.col("cluster") == cluster)
        tile_dir = paths.downscaled_tiles.joinpath(f"cluster-{cluster:02d}")
        description = descriptions[cluster]
        element = FigureElement(cluster, stats, df_tile_cluster, tile_dir, description)
        figure_elements.append(element)

    num_cpf = int(num_clusters / num_figures)
    for i in range(num_figures):
        create_figure(
            paths.output_dir,
            i,
            figure_elements[i * num_cpf : (i + 1) * num_cpf],
            variant,
        )


if __name__ == "__main__":
    main()
