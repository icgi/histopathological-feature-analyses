import datetime

import fire
import numpy, pandas
import cycler, textwrap, collections
import scipy.cluster, sklearn.cluster
import io, matplotlib, matplotlib.ticker, matplotlib.pyplot as plt


class Runner:

    # pathologist's description of superclusters produced with default parameters
    # (see `compute_and_visualize_superclusters.py run --help`)
    DESCRIPTIONS_FOR_SUBMITTED_ILLUSTRATION = {
        "G0": "predominantly benign glands or low-grade tumor with prominent lymphocytic and/or mixed lymphocytic and polymorphonuclear infiltrate and fibrocollagenous stroma",
        "G1": "high-grade mucinous tumors",
        "G2": "cellular fibroblastic stroma either as a predominant feature or in combination with high-grade tumor",
        "G3": "low-grade tumors with well-formed glands and either prominent goblet cells or medium lymphocytic infiltrate",
        "G4": "low grade tumors either with low to medium amount of immature stroma or with disintegration of tissue",
        "G5": "high-grade tumors containing nuclei with prominent nucleoli",
        "G6": "tumors with prominent necrosis",
        "G7": "low-grade tumors with more complex (cribriform and branching) growth patterns, with low lymphocytic infiltrates",
    }

    # paths to input assets (eg. original clusters' centers, tile scores) for each resolution
    PATHS = pandas.DataFrame(
        {
            "lowres": {
                "tile_clusters": "notebooks/temporary/ht-score_tile-clusters_uni_lowres_max-per-patient-100_kmeans-100.csv",
                "tile_clusters_visualization": "notebooks/temporary/pathologist_selection/ht-score_tile-clusters_pathologist-selection_lowres.csv",
                "cluster_centers": "notebooks/temporary/cluster-centres_uni_lowres_max-per-patient-100_kmeans-100.csv",
                "clusters_of_interest": "notebooks/temporary/clusters_of_interest/selected_clusters_uni_lowres_max-per-patient-100_kmeans-100.csv",
            },
            "highres": {
                "tile_clusters": "notebooks/temporary/ht-score_tile-clusters_uni_highres_max-per-patient-1000_kmeans-100.csv",
                "tile_clusters_visualization": "notebooks/temporary/pathologist_selection/ht-score_tile-clusters_pathologist-selection_highres.csv",
                "cluster_centers": "notebooks/temporary/cluster-centres_uni_highres_max-per-patient-1000_kmeans-100.csv",
                "clusters_of_interest": "notebooks/temporary/clusters_of_interest/selected_clusters_uni_highres_max-per-patient-1000_kmeans-100.csv",
            },
        }
    )

    class util:
        """ """

        @staticmethod
        def get_output_timestamp(format_="%Y-%m-%d"):
            """Provides a normalized format timestamp for marking output files"""
            return datetime.datetime.now().strftime(format_)

        @staticmethod
        def extend_default_color_cycle():
            """

            Extends default `matplotlib` color cycle to tab20

            """
            cmap = plt.get_cmap("tab20")
            colors = [cmap(i) for i in numpy.linspace(0, 1, 20)]

            matplotlib.rcParams["axes.prop_cycle"] = cycler.cycler("color", colors)

            return colors

        @staticmethod
        def plot_dendrogram(model, labels, **kwargs):
            """
            source:
            https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html

            """

            counts = numpy.zeros(model.children_.shape[0])
            n_samples = len(model.labels_)
            for i, merge in enumerate(model.children_):
                current_count = 0
                for child_idx in merge:
                    if child_idx < n_samples:
                        current_count += 1  # leaf node
                    else:
                        current_count += counts[child_idx - n_samples]
                counts[i] = current_count

            linkage_matrix = numpy.column_stack(
                [model.children_, model.distances_, counts]
            ).astype(float)

            dendrogram_parameters = scipy.cluster.hierarchy.dendrogram(
                linkage_matrix, labels=labels, **kwargs
            )

            tree = scipy.cluster.hierarchy.to_tree(linkage_matrix)

            return tree, dendrogram_parameters, linkage_matrix

        @staticmethod
        def aggregate_cluster_frame(frame):
            cluster_frame = (
                frame.groupby("cluster")["histotyping_score"]
                .agg(lambda x: x.median())
                .rename("histotyping_score_median")
            )

            cluster_frame = pandas.DataFrame(cluster_frame)
            cluster_frame["paths"] = frame.groupby("cluster")["path"].agg(
                lambda x: x.values
            )

            return cluster_frame

    def cluster_and_visualize(
        self,
        frames,
        experiment_name,
        histotyping_score_weight,
        resolution,
        linkage,
        n_clusters,
        metric="euclidean",
        **other
    ):
        """

        Groups original clusters into `n_clusters` superclusters using sklearn.cluster.AgglomerativeClustering.
        Adds tile median of `DoMore-v1-CE-CRC` for the selected `resolution` as an additional input feature.
        This `DoMore-v1-CE-CRC` feature is weighted by `histotyping_score_weight` (default for 10x is 109)

        """

        clusters_of_interest_labels = frames.loc["clusters_of_interest", resolution][
            "cluster"
        ].to_list()

        data = frames.loc["cluster_centers", resolution].drop(columns="cluster")

        cluster_frame = self.util.aggregate_cluster_frame(
            frames.loc["tile_clusters", resolution]
        )
        cluster_to_histotyping_score = cluster_frame[
            "histotyping_score_median"
        ].to_dict()

        # adds `DoMore-v1-CE-CRC score` as an additional weighted feature
        data[1024] = [
            cluster_to_histotyping_score.get(i) * histotyping_score_weight
            for i in range(0, 100)
        ]

        data = data.iloc[clusters_of_interest_labels]

        selected_to_full = frames.loc["clusters_of_interest", resolution][
            "cluster"
        ].to_dict()
        full_to_selected = {v: k for k, v in selected_to_full.items()}

        model = sklearn.cluster.AgglomerativeClustering(
            n_clusters=n_clusters,
            distance_threshold=None,
            compute_distances=True,
            metric=metric,
            linkage=linkage,
        ).fit(data.values)

        incoming_labels = clusters_of_interest_labels

        mapping = dict(zip(incoming_labels, model.labels_))

        figure, subplots = plt.subplots(1, 3, width_ratios=(0.4, 0.2, 0.4), sharey=True)
        figure.set_dpi(200)
        figure.set_size_inches(14, 12)

        subplots[0].set_xscale("log")  # ! do not move this

        output, dendrogram_parameters, linkage_matrix = self.util.plot_dendrogram(
            model,
            orientation="left",
            distance_sort=True,
            link_color_func=lambda x: "lightgray",
            labels=clusters_of_interest_labels,
            ax=subplots[0],
        )

        subplots[0].xaxis.set_major_locator(
            matplotlib.ticker.LogLocator(
                base=10.0, subs=numpy.arange(1.0, 4.0), numticks=7
            )
        )

        subplots[0].xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, y: "{}".format(x))
        )

        dendrogram_parameters = dict(dendrogram_parameters)

        dendrogram_parameters["leaves_color_list"] = [
            "C{}".format(int(mapping[int(n)])) for n in dendrogram_parameters["ivl"]
        ]

        #
        #    MARK GROUP BOUNDARIES
        #

        for leaf_n in range(len(dendrogram_parameters["ivl"])):
            leaf_name = dendrogram_parameters["ivl"][leaf_n]
            color = colors[int(mapping[int(leaf_name)])]

            subplots[0].axhspan(
                10 * leaf_n - 5 + 5,
                10 * leaf_n + 5 + 5,
                1,
                0.975,
                color=color,
                zorder=10,
            )

        boxplot_positions = []
        for leaf_n in range(len(dendrogram_parameters["ivl"])):
            boxplot_positions.append(10 * leaf_n + 5)

        for leaf_n in range(len(dendrogram_parameters["ivl"])):
            leaf_name = dendrogram_parameters["ivl"][leaf_n]
            color = colors[int(mapping[int(leaf_name)])]

            subplots[1].axhspan(
                10 * leaf_n - 4.5 + 5, 10 * leaf_n + 4.5 + 5, color=color, alpha=0.25
            )

        subplots[0].set_xlabel("Euclidean distance between clusters")

        frame_for_boxplot = frames.loc["tile_clusters", resolution]

        frame_for_boxplot = frame_for_boxplot[
            frame_for_boxplot["cluster"].isin(clusters_of_interest_labels)
        ]
        boxplot_cluster_data = frame_for_boxplot.groupby("cluster")["histotyping_score"]

        subplots[1].boxplot(
            boxplot_cluster_data.agg(list)
            .loc[[selected_to_full[i] for i in dendrogram_parameters["leaves"]]]
            .values,
            positions=boxplot_positions,
            vert=False,
            tick_labels=dendrogram_parameters["leaves"],
            flierprops={
                "markersize": 3,
                "marker": "+",
            },
            widths=3,
        )

        for tick in subplots[0].yaxis.get_majorticklabels():
            tick.set_horizontalalignment("center")

        subplots[0].tick_params(axis="both", which="major", labelsize=8)
        subplots[0].set_title("experiment: {}".format(experiment_name))
        subplots[1].tick_params(axis="both", which="major", labelsize=8)
        subplots[1].tick_params(axis="y", which="both", length=0)
        subplots[1].set_xlim(0, 1)
        subplots[1].axvline(0.5, color="black", linestyle="dashed", linewidth=1)
        subplots[1].set_xlabel("DoMore-v1-CE-CRC\n10× score")

        #
        #   APPLY LABEL COLORING
        #

        new_labels = []
        for label in subplots[0].get_ymajorticklabels():
            if label.get_text() != "":
                feature_index = int(label.get_text())
                cluster_label = mapping[selected_to_full[feature_index]]
                label.set_color(colors[cluster_label])
                new_labels.append(selected_to_full[feature_index])

        subplots[0].set_yticklabels(([""] * 30) + new_labels)

        for label in subplots[0].get_ymajorticklabels():
            feature_index = int(label.get_text())
            cluster_label = mapping[feature_index]
            label.set_color(colors[cluster_label])

        #
        #    ADD PATHOLOGIST SUPERCLUSTER ANNOTATIONS
        #

        subplots[2].axis("off")

        # enables annotations only if the parameters correspond to the version submitted for review
        if (
            n_clusters == 8
            and histotyping_score_weight == 109
            and resolution == "lowres"
            and linkage == "ward"
            and metric == "euclidean"
        ):

            wrap_and_join = lambda s: "\n".join(textwrap.wrap(s, 42))
            Runner.DESCRIPTIONS_FOR_SUBMITTED_ILLUSTRATION = {
                k: wrap_and_join(v)
                for k, v in Runner.DESCRIPTIONS_FOR_SUBMITTED_ILLUSTRATION.items()
            }

            group_midpoints = collections.defaultdict(lambda: [])
            group_colors = {}

            for leaf_n in range(len(dendrogram_parameters["ivl"])):
                leaf_name = dendrogram_parameters["ivl"][leaf_n]
                group = int(mapping[int(leaf_name)])

                color = colors[group]

                start = 10 * leaf_n - 5 + 5
                end = 10 * leaf_n + 5 + 5

                group_midpoints[group].append(start)
                group_midpoints[group].append(end)
                group_colors[group] = color

            for group, midpoints in group_midpoints.items():
                group_midpoint = min(midpoints) + (
                    (max(midpoints) - min(midpoints)) / 2
                )

                group_name = "G{}".format(group)
                group_color = group_colors[group]

                subplots[2].text(
                    s=group_name,
                    x=0.04,
                    y=group_midpoint - 1,
                    ha="left",
                    va="center",
                    color=group_color,
                    fontsize=33,
                    bbox=dict(
                        facecolor="white",
                        edgecolor=group_color,
                        boxstyle="round,pad=0.2",
                        linewidth=5,
                    ),
                )

                subplots[2].text(
                    s=Runner.DESCRIPTIONS_FOR_SUBMITTED_ILLUSTRATION[group_name],
                    x=0.26,
                    y=group_midpoint,
                    ha="left",
                    va="center",
                    fontsize=10,
                )

        plt.subplots_adjust(wspace=0.062)

        plt.savefig(
            "output___dendrogram_of_superclusters_with_score_boxplots___{}.svg".format(
                Runner.util.get_output_timestamp()
            ),
            format="svg",
        )
        plt.savefig(
            "output___dendrogram_of_superclusters_with_score_boxplots___{}.eps".format(
                Runner.util.get_output_timestamp()
            ),
            format="eps",
        )
        plt.savefig(
            "output___dendrogram_of_superclusters_with_score_boxplots___{}.png".format(
                Runner.util.get_output_timestamp()
            ),
            format="png",
        )

        # -
        # cluster_frame.drop(columns=["paths"]).to_excel("cluster_frame.xlsx")

        # serializes .svg for web presentation
        dendrogram_plot_io = io.StringIO()
        plt.savefig(dendrogram_plot_io, format="svg")
        encoded_dendrogram_plot = dendrogram_plot_io.getvalue()

        plt.show()
        plt.close(figure)

        dendrogram_frame = pandas.DataFrame(
            {
                "cluster": [
                    selected_to_full[i] for i in dendrogram_parameters["leaves"]
                ],
                "dendrogram_cluster": dendrogram_parameters["leaves_color_list"],
            }
        )

        dendrogram_frame = pandas.merge(
            frames.loc["tile_clusters", resolution].reset_index(),
            dendrogram_frame,
            left_on="cluster",
            right_on="cluster",
            how="left",
        )

        cluster_to_dendrogram_cluster = dendrogram_frame.set_index("cluster")[
            ["dendrogram_cluster"]
        ].to_dict()["dendrogram_cluster"]

        #
        #   DENDROGRAM CLASS UPDATE
        #

        dendrogram_frame = pandas.DataFrame(
            {
                "cluster": dendrogram_parameters["leaves"],
                "dendrogram_cluster": dendrogram_parameters["leaves_color_list"],
            }
        )

        dendrogram_frame = pandas.merge(
            cluster_frame.reset_index(),
            dendrogram_frame,
            left_on="cluster",
            right_on="cluster",
        )

        frames.loc["tile_clusters_visualization", resolution]["dendrogram_cluster"] = (
            frames.loc["tile_clusters_visualization", resolution]["cluster"].apply(
                cluster_to_dendrogram_cluster.get
            )
        )

        frames.loc["tile_clusters", resolution]["dendrogram_cluster"] = frames.loc[
            "tile_clusters", resolution
        ]["cluster"].apply(cluster_to_dendrogram_cluster.get)

        #
        #   REPORT
        #

        coi_with_dendrogram_cluster = pandas.merge(
            frames.loc["clusters_of_interest", resolution],
            dendrogram_frame,
            left_on="cluster",
            right_on="cluster",
        )

        coi_distribution = coi_with_dendrogram_cluster.groupby("dendrogram_cluster")[
            "cluster"
        ].agg(list)
        coi_sd = coi_distribution.apply(lambda x: len(set(x))).std()

        report = coi_distribution

        grouped_by_dendrogram_cluster = (
            frames.loc["tile_clusters_visualization", resolution]
            .sample(frac=1)
            .groupby(["dendrogram_cluster", "cluster"])
            .agg(list)["path"]
            .to_dict()
        )

        dendrogram_cluster_to_color_mapping = {
            dc: (numpy.array(matplotlib.colors.to_rgb(dc)) * 255).round(0).astype("int")
            for dc in frames.loc["tile_clusters", resolution][
                "dendrogram_cluster"
            ].unique()
            if dc is not numpy.nan
        }

        cluster_names = cluster_frame.index

        return (
            report,
            encoded_dendrogram_plot,
            grouped_by_dendrogram_cluster,
            dendrogram_cluster_to_color_mapping,
            cluster_names,
            cluster_to_histotyping_score,
        )

    def run(
        self,
        # general parameters for agglomerative clustering
        experiment_name="<experiment without name>",
        resolution="lowres",
        n_clusters=8,
        histotyping_score_weight=109,
        metric="euclidean",
        linkage="ward",
        # visualization parameters for web rendering
        # (not applicable in standalone script)
        width=2,
        height=5,
    ):
        """

        The group annotations will only be visible using the `AgglomerativeClustering`
        parameters from the submitted plot.

        Parameters and their values for the submitted plot are:

            * `n_clusters` = 8
            * `resolution` = 'lowres'
            * `histotyping_score_weight` = 109
            * `metric` = 'euclidean`
            * `linkage` = 'ward'

        The annotation text for G0 ... G7 is only valid for this parameter set.

        This parameter set is applied whenever `compute_and_visualize_superclusters.py` is
        run with default parameters:

            python3 compute_and_visualize_superclusters.py run

        """

        context = {}
        experiment_parameters = {}

        frames = Runner.PATHS.applymap(pandas.read_csv)

        experiment_parameters["experiment_name"] = experiment_name
        experiment_parameters["resolution"] = resolution
        experiment_parameters["n_clusters"] = n_clusters
        experiment_parameters["histotyping_score_weight"] = histotyping_score_weight
        experiment_parameters["metric"] = metric
        experiment_parameters["linkage"] = linkage

        context["width"] = width
        context["height"] = height

        context.update(experiment_parameters)

        (
            report,
            encoded_dendrogram_plot,
            grouped_by_dendrogram_cluster,
            dendrogram_cluster_to_color_mapping,
            cluster_names,
            cluster_to_histotyping_score,
        ) = self.cluster_and_visualize(frames=frames, **experiment_parameters)

        context["tiles"] = grouped_by_dendrogram_cluster
        context["plot"] = encoded_dendrogram_plot

        context["color_mapping"] = dendrogram_cluster_to_color_mapping
        context["cluster_names"] = cluster_names
        context["cluster_to_histotyping_score"] = cluster_to_histotyping_score


if __name__ == "__main__":
    colors = Runner.util.extend_default_color_cycle()
    fire.Fire(Runner())
