import argparse
from pathlib import Path

import polars as pl
from tqdm import tqdm


def process(input_root, output_path, pattern):
    if output_path.exists():
        print("Skipping since output path exist: {output_path}")
        return

    variant = output_path.stem.split("_")[-1]
    print(f"Search for {variant} files")
    paths = sorted(list(input_root.glob(pattern)))
    print(f"Number of input files: {len(paths)}")

    dfs = []
    for path in tqdm(paths):
        df = pl.read_csv(path).select(["filename", "ensemble"])
        df = df.with_columns(
            pl.col("filename")
            .map_elements(
                lambda p: Path(p).parents[2].name.replace("_histotyping", ""),
                return_dtype=str,
            )
            .alias("scan")
        )
        df = df.with_columns(
            pl.col("filename")
            .map_elements(lambda p: Path(p).stem.split("___")[-1], return_dtype=str)
            .alias("coordinate")
        )
        df = df.rename({"ensemble": "histotyping_score"})
        df = df.select(["scan", "coordinate", "histotyping_score"])
        dfs.append(df)
    print(f"Writing output to '{output_path}'")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pl.concat(dfs).write_csv(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        metavar="PATH",
        type=Path,
        help="Input root",
    )
    parser.add_argument(
        "output",
        metavar="PATH",
        type=Path,
        help="Output dir",
    )
    parser.add_argument(
        "prefix",
        metavar="STR",
        type=str,
        help="Common scan name prefix. Also used in output filename.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    process(
        args.input,
        args.output.joinpath(f"histotyping_ensemble_{args.prefix.lower()}-lowres.csv"),
        f"{args.prefix}*/histotyping/histotyping_tiles_lowres.csv",
    )
    process(
        args.input,
        args.output.joinpath(f"histotyping_ensemble_{args.prefix.lower()}-highres.csv"),
        f"{args.prefix}*/histotyping/histotyping_tiles_highres.csv",
    )


if __name__ == "__main__":
    main()
