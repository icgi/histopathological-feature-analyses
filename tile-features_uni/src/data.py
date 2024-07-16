from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence
from torchvision import transforms

import cv2
from torch.utils.data import Dataset


def load_tile_folders(tile_folder_list_path: Path) -> List[Path]:
    with tile_folder_list_path.open() as f:
        tile_folders = [Path(p.strip()) for p in f.readlines()]
    ok = True
    for path in tile_folders:
        if not path.exists():
            print(f"ERROR: Input tile folder does not exist: {path}")
            ok = False
    if not ok:
        print("Terminating due to non-existing input")
        exit()
    print(f"Found {len(tile_folders)} input tile folders")
    # tile_paths = [p for f in tile_folders for p in f.glob("*jpg")]
    # print(f"Found {len(tile_paths)} input tile paths")
    return tile_folders


class Tiles(Dataset):
    def __init__(self, tile_paths: Sequence[Path], transform: Callable):
        self.tile_paths = tile_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.tile_paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_filepath = str(self.tile_paths[index])
        image = cv2.imread(image_filepath, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        image = self.transform(image)
        sample = {
            "image": image,
            "height": height,
            "width": width,
            "path": image_filepath,
        }

        return sample


def tile_transform() -> Callable:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Resize(224),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transform
