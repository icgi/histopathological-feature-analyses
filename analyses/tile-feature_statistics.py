import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt
import polars as pl


def plot_bars(data, name, output_dir):
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 5)
    ax.bar(list(range(len(data))), data, width=1)
    ax.set_ylabel(name)
    ax.set_xlabel("Feature index")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir.joinpath(f"feature-statistics_{name}.png")
    plt.savefig(output_path, dpi=100, bbox_inches="tight", transparent=False)
    plt.close()


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
        help="Output directory",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    assert args.input.suffix == ".csv"
    assert args.input.is_file()

    print("Computing tile feature mean")
    start = time.time()
    query = pl.scan_csv(args.input).drop("path").mean()
    mean = query.collect(streaming=True).to_numpy()[0, :]
    plot_bars(mean, "mean", args.output)
    print(f"Elapsed: {time.time() - start:.0f} seconds")

    print("Computing tile feature standard deviation")
    start = time.time()
    query = pl.scan_csv(args.input).drop("path").std()
    std = query.collect(streaming=True).to_numpy()[0, :]
    plot_bars(std, "std", args.output)
    print(f"Elapsed: {time.time() - start:.0f} seconds")


if __name__ == "__main__":
    main()
