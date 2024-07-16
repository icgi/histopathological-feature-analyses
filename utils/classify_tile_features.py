import argparse
from pathlib import Path
import time

import numpy as np
import polars as pl
from scipy.spatial.distance import cdist


def check_tile_paths(df_class, df_feat):
    """Check that all tiles with features have a cluster class"""
    paths_class = df_class.get_column("path").unique().to_list()
    paths_feat = df_feat.get_column("path").unique().to_list()
    assert set(paths_feat).difference(paths_class) == set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tile_features",
        metavar="PATH",
        type=Path,
        help=".csv with tile features",
    )
    parser.add_argument(
        "classification",
        metavar="PATH",
        type=Path,
        help=".csv with tile cluster centres",
    )
    parser.add_argument(
        "output",
        metavar="PATH",
        type=Path,
        help=".csv with resulting classified tiles",
    )
    args = parser.parse_args()

    assert args.tile_features.is_file()
    assert args.tile_features.suffix == ".csv"
    assert args.classification.is_file()
    assert args.classification.suffix == ".csv"
    assert args.output.suffix == ".csv"

    print("Read classification file")
    df_class = pl.read_csv(args.classification)
    print(df_class)
    print("Read tile feature file")
    start = time.time()
    df_feat = pl.read_csv(args.tile_features)
    print(df_feat)
    print(f"Elapsed: {time.time() - start:.0f} s")

    feature_names = [c for c in df_class.columns if c != "cluster"]
    cluster_features = df_class.select(feature_names).to_numpy()
    tile_features = df_feat.select(feature_names).to_numpy()
    print(f"Cluster centres: {cluster_features.shape}")
    print(f"Tile features:   {tile_features.shape}")

    # matrix (shape (tiles, clusters)) with distances from every tile feature vector to
    # every cluster center vector
    dists = cdist(tile_features, cluster_features)
    print(f"Distances:       {dists.shape}")
    assert dists.shape == (tile_features.shape[0], cluster_features.shape[0])
    tile_cluster_distances = np.min(dists, axis=1)
    tile_cluster_indices = np.argmin(dists, axis=1)

    cluster_names = np.array(df_class.get_column("cluster"))
    df_result = pl.DataFrame(
        {
            "path": df_feat.get_column("path"),
            "cluster": cluster_names[tile_cluster_indices],
            "distance": tile_cluster_distances,
        }
    )

    print("Result")
    print(df_result)
    print(f"Write '{args.output}'")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_result.write_csv(args.output)


if __name__ == "__main__":
    main()
