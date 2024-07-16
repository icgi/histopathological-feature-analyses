import argparse
from pathlib import Path
import time

import polars as pl
from tqdm import tqdm


def scan_from_path(path):
    scan = Path(path).parents[2].name
    return scan


def patient_from_scan(scan):
    patient = scan[:8]
    return patient


def patient_from_path(path):
    scan = scan_from_path(path)
    return patient_from_scan(scan)


def collect_tile_features(input_dir):
    scan_feature_files = sorted(list(input_dir.rglob("*.csv")))
    print(f"Input {len(scan_feature_files)} scan feature files")

    print("Reading input csvs")
    dfs = []
    for path in tqdm(scan_feature_files):
        dfs.append(pl.read_csv(path).sort("path"))

    return pl.concat(dfs)


def subsample_per_patient(df, max_per_patient):
    df = df.with_columns(
        pl.col("path")
        .map_elements(patient_from_path, return_dtype=str)
        .alias("patient")
    )
    patients = sorted(df.get_column("patient").unique().to_list())
    dfs = []
    for patient in patients:
        df_p = df.filter(pl.col("patient") == patient)
        dfs.append(
            df_p.sample(
                n=min(len(df_p), max_per_patient),
                with_replacement=False,
                shuffle=True,
                seed=0,
            )
        )
    return pl.concat(dfs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_raw",
        metavar="PATH",
        type=Path,
        help="Input root with .csv's somewhere below",
    )
    parser.add_argument(
        "--input_result",
        metavar="PATH",
        type=Path,
        help=(
            "Input already computed csv. Typlically a complete csv to use "
            "for subsampling"
        ),
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        required=True,
        type=Path,
        help="Output csv",
    )
    parser.add_argument(
        "--max_per_patient",
        metavar="INT",
        type=int,
        help="Randomly select --max_per_patient tiles per scan for clustering",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output",
    )
    args = parser.parse_args()

    start_total = time.time()
    assert args.output.suffix == ".csv"
    if args.output.exists() and not args.overwrite:
        print(f"Output path exists, pass --overwrite to overwrite: '{args.output}'")
        exit()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.input_raw is not None:
        start_input = time.time()
        df = collect_tile_features(args.input_raw)
    elif args.input_result is not None:
        start_input = time.time()
        print(f"Reading csv from {args.input_result}")
        df = pl.read_csv(args.input_result)
    else:
        print("No input given. Terminating")
        return
    print(df)
    print(f"Get input elapsed time: {time.time() - start_input:.0f} seconds")

    if args.max_per_patient is not None:
        print(f"Selecting max {args.max_per_patient} tiles per patient")
        start_sample = time.time()
        df = subsample_per_patient(df, args.max_per_patient)
        print(df)
        print(f"Sample elapsed time: {time.time() - start_sample:.0f} seconds")

    print(f"Write output: '{args.output}'")
    df.write_csv(args.output)

    print(f"Total elapsed time: {time.time() - start_total:.0f} seconds")


if __name__ == "__main__":
    main()
