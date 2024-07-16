import argparse
from pathlib import Path
import time

import polars as pl
from sklearn import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data",
        metavar="PATH",
        type=Path,
        help="Input .csv with features",
    )
    parser.add_argument(
        "result",
        metavar="PATH",
        type=Path,
        help="Input .csv with cluster result",
    )
    parser.add_argument(
        "--silhouette",
        action="store_true",
        help="Also perform the more expensive silhouette analysis",
    )
    args = parser.parse_args()

    assert args.data.is_file()
    assert args.data.suffix == ".csv"
    assert args.result.is_file()
    assert args.data.suffix == ".csv"

    print("Reading data")
    start = time.time()
    df_data = pl.read_csv(args.data)
    print(df_data)
    print(f"Elapsed: {time.time() - start:.0f} s")

    print("Reading result")
    start = time.time()
    df_result = pl.read_csv(args.result)
    print(df_result)
    print(f"Elapsed: {time.time() - start:.0f} s")

    data = df_data.drop("path").to_numpy()
    labels = df_result.get_column("cluster").to_numpy()

    result = {}

    if args.silhouette:
        print("Silhouette analysis")
        start = time.time()
        result["Sihouette score"] = metrics.silhouette_score(data, labels)
        print(list(result.items())[-1])
        print(f"Elapsed: {time.time() - start:.0f} s")

    print("Calinski-Harabasz analysis")
    start = time.time()
    result["Calinski-Harabasz index"] = metrics.calinski_harabasz_score(data, labels)
    print(list(result.items())[-1])
    print(f"Elapsed: {time.time() - start:.0f} s")

    print("Davies-Bouldin analysis")
    start = time.time()
    result["Davies-Bouldin index"] = metrics.davies_bouldin_score(data, labels)
    print(list(result.items())[-1])
    print(f"Elapsed: {time.time() - start:.0f} s")

    print("Summary")
    for k, v in result.items():
        print(f"{k:<30}{v:9.4f}")


if __name__ == "__main__":
    main()
