import argparse
from pathlib import Path
from typing import Any, List, Sequence

from tqdm import tqdm


def filter_existing(
    existing: Sequence[Path], tile_folders: Sequence[Path]
) -> List[Path]:
    print("Filtering out tile folders from existing files")
    existing_tile_folders = []
    for existing_path in existing:
        print(f"Reading '{existing_path}'")
        with existing_path.open() as f:
            existing_tile_folders.extend(
                [Path(r.strip()) for r in f.readlines() if Path(r.strip()).exists()]
            )
    tile_folders = sorted(list(set(tile_folders).difference(existing_tile_folders)))
    return tile_folders


def split_list(lst: Sequence[Any], num_parts: int) -> List[Any]:
    part_len, rem = divmod(len(lst), num_parts)
    return [
        lst[i * part_len + min(i, rem) : (i + 1) * part_len + min(i + 1, rem)]
        for i in range(num_parts)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        metavar="PATH",
        type=Path,
        help="Input root dir",
    )
    parser.add_argument(
        "version",
        metavar="STR",
        type=str,
        choices=["highres", "lowres"],
        help="Highres or lowres",
    )
    parser.add_argument(
        "output",
        metavar="PATH",
        type=Path,
        help="Output txt file",
    )
    parser.add_argument(
        "--existing",
        metavar="PATH",
        type=Path,
        nargs="+",
        help="Existing files. Only write tile folders not in these",
    )
    parser.add_argument(
        "--split",
        metavar="INT",
        type=int,
        help="How many parts to split the result into",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output csv files",
    )
    args = parser.parse_args()

    assert args.output.parent.exists()
    assert args.output.suffix == ".txt"

    if args.output.exists() and not args.overwrite:
        print(f"Output exist, pass --overwrite to overwrite: '{args.output}'")
        exit()

    tile_folders = sorted(list(args.input.glob(f"*/tiling/{args.version}_20x/")))
    print("Initial tile folders:", len(tile_folders))

    if args.existing is not None:
        tile_folders = filter_existing(args.existing, tile_folders)
        print("After filtering existing tile folders:", len(tile_folders))

    non_empty_tile_folders = []
    for tile_folder in tqdm(tile_folders):
        if len(list(tile_folder.glob("*png"))) > 0:
            non_empty_tile_folders.append(tile_folder)

    tile_folders = non_empty_tile_folders
    print("Non-empty tile folders:", len(tile_folders))

    if args.split:
        print(f"Split input into {args.split} splits")
        parts = split_list(tile_folders, args.split)
        for i, part in enumerate(parts):
            output_path = args.output.with_name(
                f"{args.output.stem}_part-{i + 1}-of-{args.split}.txt"
            )
            with output_path.open("w") as f:
                for tile_folder in part:
                    f.write(f"{str(tile_folder)}\n")
            print(f"Output written to '{output_path}'")

    with args.output.open("w") as f:
        for tile_folder in tile_folders:
            f.write(f"{str(tile_folder)}\n")

    print(f"Output written to '{args.output}'")


if __name__ == "__main__":
    main()
