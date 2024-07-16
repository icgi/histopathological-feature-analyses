import argparse
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


class Config:

    def __init__(self, args):
        self.mpp_lvl0_x = 0.23016019149327932  # 40x
        self.mpp_lvl0_y = 0.23016548898658135  # 40x
        self.mpp_tile_x = self.mpp_lvl0_x * 2  # 20x
        self.mpp_tile_y = self.mpp_lvl0_y * 2  # 20x
        self.mpp_base = 16.3745
        self.factor = 1.0
        self.num_clusters = 100
        self.circle = args.resolution == "lowres"
        self.cmap = "RdYlBu_r"
        self.resolution = args.resolution
        self.variant = args.variant
        self.cluster_order = "percentile-50"  # If not 'random', get from file
        self.excluded_clusters = args.exclude  # if variant == 'mask'


def write_scale_bar(output_path, mpp, width_mm, thickness=10):
    """
    Example:
    mpp: 5 µm/pixel
    width_mm: 5 mm
    5 µm/pixel * 200 pixels = 5 mm:

    rule = np.zeros((thickness, 250))
    """

    width_pixels = int(1000 * width_mm / mpp)

    rule = np.zeros((thickness, width_pixels))
    cv2.imwrite(str(output_path), rule)


def compute_contour_thickness(image_shape, input_value):
    if input_value is None:
        height = image_shape[0]
        width = image_shape[1]
        diagonal = np.sqrt(height * height + width * width)
        thickness = max(1, int(np.round(diagonal / 1000)))
    else:
        thickness = input_value
    return thickness


def draw_contour(image, mask, color=[255, 255, 255], thickness=None, connectivity=8):
    """Draw contours of the mask on the image

    NOTE: This will delineate both inside and outside the mask.
    """
    thickness = compute_contour_thickness(image.shape, thickness)
    mask_contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for cnt in mask_contours:
        cv2.drawContours(
            image,
            [cnt],
            0,
            color=color[::-1],
            thickness=thickness,
            lineType=connectivity,
        )

    return image


def resize_dim(image, factor):
    height = image.shape[0]
    width = image.shape[1]
    return (round(width * factor), round(height * factor))


def resize_image(image, dim=None, factor=None, binary=False):
    if dim is None:
        assert factor is not None
        dim = resize_dim(image, factor)
    if binary:
        image = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
        image = ((image > 127) * 255).astype(np.uint8)
    else:
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return image


def create_overlay(df, base, seg, config, cluster_colors=None):
    base = resize_image(base, factor=config.factor)
    seg = resize_image(seg, dim=(base.shape[1], base.shape[0]), binary=True)
    overlay = draw_contour(base, seg, color=[255, 255, 255], thickness=16)
    overlay = draw_contour(overlay, seg, color=[0, 0, 0], thickness=4)

    if config.resolution == "lowres":
        size_tile = 1024
    elif config.resolution == "highres":
        size_tile = 256
    else:
        print("ERROR: Invalid resolution value")
    mpp_target = config.mpp_base / config.factor
    size_target_x = size_tile * config.mpp_tile_x / mpp_target
    size_target_y = size_tile * config.mpp_tile_y / mpp_target

    if config.variant == "mask":
        df = df.filter(pl.col("include"))

    for row in df.iter_rows(named=True):
        x = row["x"]
        y = row["y"]
        if config.variant == "cluster":
            cluster = row["cluster"]
            color = cluster_colors[cluster]
        elif config.variant in ["score", "mask"]:
            value = row["histotyping_score"]
            color = mpl.colormaps[config.cmap](value)[:3]
        # elif config.variant == "mask":
        #     color = [77, 187, 213] if row["include"] else [230, 75, 53]
        #     color = np.array(color) / 255
        else:
            print(f"Invalid variant: '{config.variant}'")
            exit()
        color = list(np.array(color[::-1]) * 255)
        if config.circle:
            r = int((size_target_x + size_target_y) / 2 / 2)
            cv2.circle(overlay, (int(x), int(y)), r, color, thickness=-1)
        else:
            s_x = size_target_x / 2
            s_y = size_target_x / 2
            cv2.rectangle(
                overlay,
                (int(x - s_x), int(y - s_y)),
                (int(x + s_x), int(y + s_y)),
                color,
                thickness=-1,
            )

    if config.variant == "cluster":
        included_clusters = df.get_column("cluster").unique().to_list()
        included_colors = {
            k: v for k, v in cluster_colors.items() if k in included_clusters
        }
    else:
        included_colors = None

    return overlay, included_colors


def add_coordinates(df, config):
    """
    'coordinate' in input df is on the form 'x=45924_y=9612', and can be found at the
    end of the tile filename, e.g.

    lowres_20x___tile=000000___x=45924_y=9612.png

    (x, y) are tile centre points in a flipped cartesian system:
    x: col in original MPP from left to right
    y: row in original MPP from top to bottom
    """
    if "coordinate" not in df.columns:
        assert "path" in df.columns
        df = df.with_columns(
            pl.col("path")
            .map_elements(lambda p: Path(p).stem.split("___")[-1], return_dtype=str)
            .alias("coordinate")
        )
    mpp_target = config.mpp_base / config.factor

    def scale_x_coord(c):
        return c * config.mpp_lvl0_x / mpp_target

    def scale_y_coord(c):
        return c * config.mpp_lvl0_y / mpp_target

    df = df.with_columns(
        pl.col("coordinate")
        .map_elements(
            lambda s: scale_x_coord(int(s.split("_")[0].split("=")[1])),
            return_dtype=float,
        )
        .alias("x")
    )
    df = df.with_columns(
        pl.col("coordinate")
        .map_elements(
            lambda s: scale_y_coord(int(s.split("_")[1].split("=")[1])),
            return_dtype=float,
        )
        .alias("y")
    )
    return df


def create_cluster_colors(
    cmap, cluster_order="random", num_clusters=None, cluster_order_path=None
):
    """Assumes clusters are named as integers from 0 to number of clusters"""
    if cluster_order == "random":
        # Cluster names are randomly assigned to clusters, therefore this is random
        cluster_order = list(range(num_clusters))
    else:
        assert cluster_order_path is not None
        df = pl.read_csv(cluster_order_path)
        print("Create cluster order from 'median' in:")
        print(df)
        cluster_order = df.sort("percentile-50").get_column("cluster").to_list()
        num_clusters = len(cluster_order)

    # named_colors = mc.CSS4_COLORS
    cluster_colors = {
        c: mpl.colormaps[cmap](i / (num_clusters - 1))[:3]
        for i, c in enumerate(cluster_order)
    }
    return cluster_colors


def write_discrete_color_legend(output_path, colors, num_cols=1):
    """
    Adapted from

    https://matplotlib.org/stable/gallery/color/named_colors.html#sphx-glr-gallery-color-named-colors-py
    """
    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12
    dpi = 100

    num_rows = np.ceil(len(colors) / num_cols)

    width = cell_width * num_cols + 2 * margin
    height = cell_height * num_rows + 2 * margin

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(
        margin / width,
        margin / height,
        (width - margin) / width,
        (height - margin) / height,
    )
    ax.set_xlim(0, cell_width * num_cols)
    ax.set_ylim(cell_height * (num_rows - 0.5), -cell_height / 2.0)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, (name, color) in enumerate(reversed(colors.items())):
        row = i % num_rows
        col = i // num_rows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(
            text_pos_x,
            y,
            f"Cluster {name}",
            fontsize=14,
            horizontalalignment="left",
            verticalalignment="center",
        )

        ax.add_patch(
            mpl.patches.Rectangle(
                xy=(swatch_start_x, y - 9),
                width=swatch_width,
                height=18,
                facecolor=color,
                edgecolor="0.7",
            )
        )

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", transparent=False)


def write_continuous_color_legend(output_path, cmap, axis=False):
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
    plt.savefig(output_path, dpi=200, bbox_inches="tight", transparent=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cluster",
        metavar="PATH",
        type=Path,
        help="Csv with tile paths, cluster and histotyping score info",
    )
    parser.add_argument(
        "histotyping",
        metavar="PATH",
        type=Path,
        nargs="+",
        help=(
            "Histotyping inference root path(s) that defines scans to display\n"
            "Expect to end with 'scan' and contain images:\n"
            "segmentation/{output_downsampled_image.png,out_combined_segmentation.png}"
        ),
    )
    parser.add_argument(
        "resolution",
        metavar="STR",
        type=str,
        choices=["lowres", "highres"],
        help="'lowres' or 'highres'",
    )
    parser.add_argument(
        "variant",
        metavar="STR",
        type=str,
        choices=["score", "cluster", "mask"],
        help="What to make heatmap of, histotyping 'score' or 'cluster' membership",
    )
    parser.add_argument(
        "--exclude",
        metavar="INT",
        type=int,
        nargs="+",
        help="Clusters to exclude if --variant == 'mask'",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        type=Path,
        help="Output root. Parent of --cluster/wsi-cluster-overlays if not given",
    )
    args = parser.parse_args()

    config = Config(args)
    if args.output is None:
        output_root = args.cluster.parent.joinpath("wsi-cluster-overlays")
    else:
        output_root = args.output
    output_root.mkdir(parents=True, exist_ok=True)

    included_scans = [p.name for p in args.histotyping]
    print(f"Including {len(included_scans)} scans")

    print(f"Extract info from {args.cluster} for included scans")
    query = pl.scan_csv(args.cluster).filter(pl.col("scan").is_in(included_scans))
    df = query.collect(streaming=True)
    df = add_coordinates(df, config)
    print(df)

    write_scale_bar(
        output_root.joinpath("scale_bar.png"),
        config.mpp_base / config.factor,  # MPP
        5,  # millimetre
        thickness=10,  # pixels
    )

    if config.variant == "cluster":
        cluster_order_path = args.cluster.with_name(args.cluster.stem + "_summary.csv")
        cluster_colors = create_cluster_colors(
            config.cmap, config.cluster_order, config.num_clusters, cluster_order_path
        )
        write_discrete_color_legend(
            output_root.joinpath("cluster_color_legend_all.png"), cluster_colors, 2
        )
    elif config.variant == "score":
        cluster_colors = None
        write_continuous_color_legend(
            output_root.joinpath("score_color_legend_all.png"),
            config.cmap,
            axis=False,
        )
    elif config.variant == "mask":
        cluster_colors = None
        df = df.with_columns(
            pl.col("cluster")
            .map_elements(
                lambda c: c not in config.excluded_clusters, return_dtype=bool
            )
            .alias("include")
        )
    else:
        print(f"Invalid variant: '{config.variant}'")
        exit()

    for scan_path in args.histotyping:
        scan = scan_path.name
        base_path = scan_path.joinpath("segmentation", "output_downsampled_image.png")
        seg_path = scan_path.joinpath("segmentation", "out_combined_segmentation.png")

        base_im = cv2.imread(str(base_path), cv2.IMREAD_COLOR)
        seg_im = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
        df_s = df.filter(pl.col("scan") == scan)
        overlay_im, included_colors = create_overlay(
            df_s, base_im, seg_im, config, cluster_colors
        )

        output_dir = output_root.joinpath(scan)
        output_dir.mkdir(exist_ok=True)
        if config.variant == "cluster":
            write_discrete_color_legend(
                output_dir.joinpath(f"color_legend_{scan}.png"), included_colors
            )
        output_path = output_dir.joinpath(f"{scan}_{config.variant}_overlay.png")
        print(f"Write '{output_path}'")
        cv2.imwrite(str(output_path), overlay_im)


if __name__ == "__main__":
    main()
