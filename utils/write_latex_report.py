import argparse
from pathlib import Path

import cv2
import polars as pl


class Name:

    def __init__(self, filename):
        """
        Example:
        uni_highres_max-per-patient-1000_kmeans-100
        """
        self.network = filename.split("_")[0]
        self.resolution = filename.split("_")[1]
        if "max-per-patient" in filename:
            max_per_patient = filename.split("_")[2].split("-")[-1]
            self.tiles_per_patient = f"max {max_per_patient}"
            method = filename.split("_")[3]
        else:
            self.tiles_per_patient = "all"
            method = filename.split("_")[2]
        self.method = method.split("-")[0]
        if self.method == "kmeans":
            self.num_clusters = method.split("-")[-1]


class Content:

    def __init__(self, root):
        self.root_path = root
        self.filename = root.name
        self.name = Name(self.filename)
        self.cluster_stats_path = root.joinpath(
            f"ht-score_tile-clusters_{self.filename}_summary.csv"
        )
        self.distance_figure_path = root.joinpath(
            "per-cluster_distance-distribution.png"
        )
        self.score_figure_path = root.joinpath("per-cluster_ht-score-distribution.png")
        self.tile_dir = root.joinpath("example-tiles_per-cluster-20_per-scan-1")
        self.scan_distribution_dir = root.joinpath("scan-cluster-distribution")

        assert root.joinpath(f"tile-clusters_{self.filename}.csv").is_file()
        assert root.joinpath(f"ht-score_tile-clusters_{self.filename}.csv").is_file()
        assert self.cluster_stats_path.is_file()
        assert self.distance_figure_path.is_file()
        assert self.score_figure_path.is_file()
        assert self.tile_dir.is_dir()
        assert self.scan_distribution_dir.is_dir()

        self.output_root = root.joinpath("report")
        self.output_root.mkdir(exist_ok=True)
        self.output_root.joinpath("temps").mkdir(exist_ok=True)

        self.output_figures_dir = self.output_root.joinpath("figures")
        self.output_figures_dir.mkdir(exist_ok=True)


def check_input(root):
    """
    For <name> = root.name, this is the expected content
        - tile-clusters_<name>.csv
        - ht-score_tile-clusters_<name>.csv
        - ht-score_tile-clusters_<name>_summary.csv
        - per-cluster_distance-distribution.png
        - per-cluster_ht-score-distribution.png
        - example-tiles_per-cluster-20_per-scan-1
          - cluster_00
          ...
          - cluster_nn
        - scan-cluster-distribution
    """
    assert root.is_dir()

    content = Content(root)

    return content


def write_main(content):
    title = f"Feature extractor:  {content.name.network}\\\\\n"
    title += f"Resolution:         {content.name.resolution}\\\\\n"
    title += f"Tiles per patient:  {content.name.tiles_per_patient}\\\\\n"
    title += f"Clustering method:  {content.name.method}\\\\\n"
    title += f"Number of clusters: {content.name.num_clusters}\\\\\n"
    title += "Path:               \\tiny{"
    title += str(content.root_path).replace("_", "\\_")
    title += "}"

    s = """\\documentclass[8pt, compress, aspectratio=169]{beamer}
\\input{config}

"""
    s += "\\title{" + title + "}"
    s += """
\\defbeamertemplate*{title page}{customized}[1][]{%
  \\usebeamerfont{title}\\inserttitle\\par
}

\\begin{document}

\\begin{frame}[plain]
  \\maketitle
\\end{frame}

\\input{content}

\\end{document}"""
    with content.output_root.joinpath("main.tex").open("w") as f:
        f.write(s)


def write_config(content):
    s = """\\usepackage[english]{babel}
\\usepackage{graphicx}
\\usepackage[utf8]{inputenc}
\\usepackage{booktabs}
\\usepackage[style=iso]{datetime2}
\\usepackage{tikz}
\\usepackage[font=footnotesize,figurename=,tablename=]{caption}
\\usepackage{subcaption}
\\usepackage{multirow}
\\usepackage{makecell}
\\usepackage{verbatim}
\\usepackage[export]{adjustbox}

\\PassOptionsToPackage{height=1cm}{beamerouterthemesidebar}
\\usetheme{Pittsburgh}
\\usecolortheme{dove}

\\usefonttheme{professionalfonts}

\\setbeamertemplate{footline}[frame number]{}
\\beamertemplatenavigationsymbolsempty%"""
    with content.output_root.joinpath("config.tex").open("w") as f:
        f.write(s)


def generate_figure(path, width=""):
    s = """  \\begin{figure}
    \\centering
"""
    s += "    \\includegraphics[width=" + width + "\\textwidth]{" + str(path) + "}"
    s += """
  \\end{figure}
"""
    return s


def generate_subfigure(path_rows):
    if len(path_rows[0]) == 1:
        width = ""
    else:
        width = str(1 / len(path_rows[0]) - 0.01)
    s = """
  \\begin{figure}[h]
    \\centering
"""
    for i, path_row in enumerate(path_rows):
        for j, path in enumerate(path_row):
            s += "    \\begin{subfigure}[t]{" + width + "\\textwidth}\n"
            s += "      \\includegraphics[width=\\textwidth]{" + str(path) + "}"
            s += """
    \\end{subfigure}"""
            if j < len(path_row) - 1:
                s += """
    %
"""
        if i < len(path_rows) - 1:
            s += "\n\n"
    s += """
  \\end{figure}

"""
    return s


def generate_overview(content):
    s = "\\begin{frame}\n"
    s += generate_subfigure(
        [[str(content.score_figure_path)], [str(content.distance_figure_path)]]
    )
    s += "\\end{frame}\n"
    return s


def copy_examples(input_dir, output_dir):
    max_included = 16
    all_input_paths = list(input_dir.glob("*png"))
    num_included = min(len(all_input_paths), max_included)
    input_paths = all_input_paths[:num_included]
    output_dir.mkdir(exist_ok=True)
    output_paths = []
    for input_path in input_paths:
        image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (200, 200), interpolation=cv2.INTER_NEAREST)
        output_path = output_dir.joinpath(f"{input_path.stem}.jpg")
        cv2.imwrite(str(output_path), image)
        output_paths.append(output_path)

    return output_paths


def generate_cluster(content, cluster, title):
    s = """
% -----------------------------------------------------------------------------
\\begin{frame}
    \\frametitle{"""
    s += title + "}"

    example_paths = copy_examples(
        content.tile_dir.joinpath(f"cluster-{cluster:02d}"),
        content.output_figures_dir.joinpath(f"cluster-{cluster:02d}"),
    )

    path_rows = [["example-image"] * 8, ["example-image"] * 8]
    for i, path in enumerate(example_paths):
        row, col = divmod(i, 8)
        path_rows[row][col] = path
    s += generate_subfigure(path_rows)

    s += generate_figure(
        content.scan_distribution_dir.joinpath(
            f"scan-percentage_cluster-{cluster:02d}.png"
        )
    )

    s += """
\\end{frame}
"""
    return s


def write_content(content):
    s = generate_overview(content)

    df_cluster_stats = pl.read_csv(content.cluster_stats_path)
    # clusters = df_cluster_stats.get_column("cluster").to_list()

    for i, row in enumerate(df_cluster_stats.iter_rows(named=True)):
        cluster = row["cluster"]
        title = f"Cluster {cluster}, "
        title += f"Tiles: {row['size']}, "
        title += f"HT mean: {row['mean']:.2f}, "
        title += f"HT variance: {row['variance']:.3f}"
        print(i, title)
        s += generate_cluster(content, cluster, title)

    with content.output_root.joinpath("content.tex").open("w") as f:
        f.write(s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        metavar="PATH",
        type=Path,
        help="Path to input folder",
    )
    args = parser.parse_args()

    # TODO: Replace with pylatex

    input_root = args.input
    content = check_input(input_root)

    write_main(content)
    write_config(content)
    write_content(content)


if __name__ == "__main__":
    main()
