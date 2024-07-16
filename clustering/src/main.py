import argparse
from pathlib import Path
import time

import polars as pl

import kmeans
import hdbscan


def format_time(total_seconds):
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    if hours > 0:
        formatted = f"{hours}h {minutes:>2}m {seconds:>2}s"
    elif minutes > 0:
        formatted = f"{minutes:>2}m {seconds:2}s"
    elif seconds > 0:
        formatted = f"{seconds:>2}s"
    else:
        formatted = f"{total_seconds:.2f}s"
    return formatted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        metavar="PATH",
        type=Path,
        help="Already collected tile features as one .csv",
    )
    parser.add_argument(
        "output",
        metavar="PATH",
        type=Path,
        help="Output csv",
    )
    parser.add_argument(
        "--method",
        metavar="STR",
        type=str,
        choices=["kmeans", "hdbscan"],
        default="kmeans",
        help="Clustering method",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()
    print("Clustering")

    assert args.input.suffix == ".csv"
    assert args.output.suffix == ".csv"
    if args.output.exists():
        print("ERROR: Output path exists:", args.output)
        exit()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading '{args.input}'")
    df = pl.read_csv(args.input)

    all_paths = df.get_column("path").to_list()
    all_features = df.drop(["path"]).to_numpy()

    start_clustering = time.time()
    if args.method == "kmeans":
        clusters, dists = kmeans.kmeans(all_features, 100)
        # kmeans.silhouette_analysis(data)
    elif args.method == "hdbscan":
        clusters, dists = hdbscan.hdbscan(all_features, 100)
    else:
        print(f"ERROR: Invalid clustering method '{args.method}'")
        exit()
    print(f"Completed clustering: {format_time(time.time() - start_clustering)}")
    df = pl.DataFrame({"path": all_paths, "cluster": clusters, "distance": dists})

    df.write_csv(args.output)


if __name__ == "__main__":
    main()
