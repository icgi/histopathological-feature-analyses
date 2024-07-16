import colorsys
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np


def change_lightness(color, shift=0, factor=1):
    try:
        c = mc.cnames[color]
    except Exception:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    lightness = max(0, min(1, c[1] * factor + shift))
    return colorsys.hls_to_rgb(c[0], lightness, c[2])


def remove_spines(ax, sides):
    """
    Remove spines of axis.

    Parameters:
      ax: axes to operate on
      sides: list of sides: top, left, bottom, right

    Examples:
    removespines(ax, ['top'])
    removespines(ax, ['top', 'bottom', 'right', 'left'])
    """
    for side in sides:
        ax.spines[side].set_visible(False)
    return ax


def change_borders(ax, linewidth=1, xtick_rotation=90):
    # Set axis line width
    [border.set_linewidth(linewidth) for border in ax.spines.values()]
    # Removes shadow outside line
    [border.set_antialiased(False) for border in ax.spines.values()]
    # Remove right and top axes
    remove_spines(ax, ["right", "top"])
    # Set tick width
    ax.yaxis.set_tick_params(width=linewidth)
    ax.xaxis.set_tick_params(width=linewidth, rotation=xtick_rotation)
    # Set tick positions
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")


def plot_violins(
    output_path,
    data,
    color="#bc3c29",
    title=None,
    xlabel=None,
    ylabel=None,
    xticklabels=None,
    y_max=1,
    fig_height=None,
    fig_width=None,
    save_svg=False,
):
    dpi = 200
    iqr_width = 0.15  # = IQR bar width / 2
    scatter_width = 0.05  # = scatter random range
    rng = np.random.default_rng()

    if xticklabels is not None:
        assert len(xticklabels) == len(data)
    else:
        xticklabels = [str(x) for x in range(1, len(data) + 1)]

    if isinstance(color, str):
        colors = [color] * len(xticklabels)
    else:
        assert isinstance(color, list)
        colors = color
    assert len(colors) == len(xticklabels)

    fig, ax = plt.subplots()
    if fig_height is not None:
        fig.set_figheight(fig_height)
    if fig_width is not None:
        fig.set_figwidth(fig_width)

    parts = ax.violinplot(
        data,
        vert=True,
        widths=0.8,
        points=100,
        bw_method="scott",
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for i, pc in enumerate(parts["bodies"]):
        # pc.set_facecolor(lighten(colors[i]))
        pc.set_facecolor(change_lightness(colors[i], 0, 1.2))
        pc.set_edgecolor(colors[i])
        pc.set_alpha(1)

    for i, vals in enumerate(data):
        pos = i + 1
        # Scatter
        # All markers on 'pos' line
        # x = np.ones_like(vals) * pos
        # Some fluctuation around 'pos' line
        x = rng.uniform(pos - scatter_width, pos + scatter_width, len(vals))
        ax.scatter(x, vals, s=10, c="black", marker=".", alpha=0.4, linewidths=0)
        # Interquartile range (vertical bar)
        q1, median, q3 = np.percentile(vals, [25, 50, 75])
        ax.fill_between(
            [pos - iqr_width, pos + iqr_width],
            q1,
            q3,
            # color=lighten(colors[i], 0.2),
            color=change_lightness(colors[i], 0, 1.5),
            linewidth=0,
        )
        # Mean and median marks (horizontal bar)
        mean = np.mean(vals)
        ax.hlines(
            mean, pos - iqr_width, pos + iqr_width, colors="black", linestyles="solid"
        )
        ax.hlines(
            median,
            pos - iqr_width,
            pos + iqr_width,
            colors=colors[i],
            linestyles="solid",
        )

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(1, len(xticklabels) + 1), labels=xticklabels)
    ax.set_xlim(0.25, len(xticklabels) + 0.75)
    ax.set_ylim([0, y_max])

    if title is not None:
        ax.set_title(title)

    change_borders(ax, 1, 0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Write '{output_path}'")
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


def plot_boxes(
    output_path,
    data,
    color="#bc3c29",
    title=None,
    xlabel=None,
    ylabel=None,
    xticklabels=None,
    y_max=1,
    fig_height=None,
    fig_width=None,
    save_svg=False,
):
    dpi = 200
    width = 0.8  # = IQR bar width / 2
    rng = np.random.default_rng()

    if xticklabels is not None:
        assert len(xticklabels) == len(data)
    else:
        xticklabels = [str(x) for x in range(1, len(data) + 1)]

    if isinstance(color, str):
        colors = [color] * len(xticklabels)
    else:
        assert isinstance(color, list)
        colors = color
    assert len(colors) == len(xticklabels)

    fig, ax = plt.subplots()
    if fig_height is not None:
        fig.set_figheight(fig_height)
    if fig_width is not None:
        fig.set_figwidth(fig_width)

    bp = plt.boxplot(
        data,
        widths=width,
        whis=[10, 90],
        showmeans=False,
        showfliers=False,
        meanline=False,
        patch_artist=True,
        vert=True,
    )
    for i, c in enumerate(colors):
        plt.setp(bp["boxes"][i], color=c, linewidth=1)
        plt.setp(bp["medians"][i], color=c, linewidth=1)
        # plt.setp(bp["means"][i], color="k", linewidth=0.5)

        lighter_c = change_lightness(c, 0, 1.5)
        bp["boxes"][i].set_facecolor(lighter_c)

    for i, vals in enumerate(data):
        pos = i + 1
        # Some fluctuation around 'pos' line
        x = rng.uniform(pos - width / 4, pos + width / 4, len(vals))
        ax.scatter(x, vals, s=10, c="black", marker=".", alpha=0.4, linewidths=0)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(1, len(xticklabels) + 1), labels=xticklabels)
    ax.set_xlim(0.25, len(xticklabels) + 0.75)
    # ax.set_ylim([0, y_max])
    # ax.axis("off")

    if title is not None:
        ax.set_title(title)

    change_borders(ax, 1, 0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Write '{output_path}'")
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
