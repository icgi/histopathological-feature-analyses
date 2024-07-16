from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data import Tiles, tile_transform
from configurations import Config
import polars as pl


def common_path(paths: Sequence[Path]) -> Optional[Path]:
    common_parent = None
    if len(paths) == 0:
        return None
    elif len(paths) == 1:
        if paths[0].is_file():
            common_parent = paths[0].parent
        else:
            common_parent = paths[0]
    else:
        common_parent = paths[0]
        for path in paths[1:]:
            if common_parent is not None:
                common_parent = common(common_parent, path)

    return common_parent


def common(path_1: Path, path_2: Path) -> Optional[Path]:
    path = None
    for component_1, component_2 in zip(path_1.parts, path_2.parts):
        if component_1 == component_2:
            if path is None:
                path = Path(component_1)
            else:
                path = path.joinpath(component_1)
        else:
            break
    return path


def extract_features(
    dataset: Dataset, model: torch.nn.Module, conf: Config
) -> pl.DataFrame:
    dataloader = DataLoader(
        dataset,
        batch_size=conf.batch_size,
        shuffle=False,
        num_workers=conf.num_workers,
        prefetch_factor=2,
        # pin_memory=True,
        # pin_memory_device=conf.device,
        drop_last=False,
    )

    records = []
    batch_pbar = tqdm(total=len(dataloader), position=0, leave=False)
    for sample_batch in dataloader:
        embeddings = model(sample_batch["image"].to(conf.device)).cpu().numpy()
        for i, path in enumerate(sample_batch["path"]):
            record = {"path": str(path)}
            record.update(
                {str(j): embeddings[i, j] for j in range(embeddings.shape[1])}
            )
            records.append(record)
        batch_pbar.update()
    df = pl.DataFrame(records)
    return df


def extract_per_folder(
    output_root: Path,
    tile_folders: Sequence[Path],
    model: torch.nn.Module,
    conf: Config,
    tile_suffix: str,
    overwrite: bool,
):
    input_root = common_path(tile_folders)
    folder_pbar = tqdm(total=len(tile_folders), position=1, leave=True)
    for tile_folder in tile_folders:
        output_path = output_root.joinpath(
            tile_folder.relative_to(input_root)
        ).with_suffix(".csv")

        if output_path.exists() and not overwrite:
            # TODO: Check on per-tile basis for existing csvs?
            folder_pbar.write(
                f"Output exist, pass --overwrite to write: '{output_path}'"
            )
            folder_pbar.update()
            continue

        tile_paths = sorted(list(tile_folder.glob(f"*{tile_suffix}")))
        folder_pbar.write(f"Input: '{tile_folder}'. Tiles: {len(tile_paths):>5}")

        dataset = Tiles(tile_paths, tile_transform())
        df = extract_features(dataset, model, conf)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(output_path)

        folder_pbar.update()
