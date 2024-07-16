import argparse
import os
from pathlib import Path

from huggingface_hub import login, hf_hub_download
import torch
import timm

import configurations
import feature_extraction
import data


def load_model(model_path: Path) -> torch.nn.Module:
    model_name = "pytorch_model.bin"
    if model_path.exists() and (
        model_path.is_file() or model_path.joinpath(model_name).exists()
    ):
        if model_path.is_dir():
            model_path = model_path.joinpath(model_name)
        print(f"Loading model from {model_path}")
    else:
        if model_path.is_dir():
            model_path = model_path.joinpath(model_name)
        else:
            assert model_path.name == model_name
        model_path.parent.mkdir(parents=True, exist_ok=True)
        login()  # User Access Token, found at https://huggingface.co/settings/tokens
        hf_hub_download(
            "MahmoodLab/UNI",
            filename=model_name,
            local_dir=model_path.parent,
            force_download=True,
        )

    # Config below is from UNI readme
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        metavar="PATH",
        type=Path,
        help="Path to model (format '<path>/pytorch_model.bin')",
    )
    parser.add_argument(
        "input",
        metavar="PATH",
        type=Path,
        help="Path to input csv with tile folders",
    )
    parser.add_argument(
        "output",
        metavar="PATH",
        type=Path,
        help="Output root folder",
    )
    parser.add_argument(
        "--gpu",
        metavar="INT",
        type=int,
        help="GPU device index to run on",
    )
    parser.add_argument(
        "--suffix",
        metavar="STR",
        type=str,
        default=".png",
        choices=[".png", ".jpg"],
        help="Tile image file suffix",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output csv files",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        assert 0 <= args.gpu < torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    conf = configurations.Config(args.gpu)

    print(f"Start feature extraction on device '{conf.device}'")

    tile_folders = data.load_tile_folders(args.input)
    model = load_model(args.model)
    model.to(conf.device)
    model.eval()

    with torch.inference_mode():
        feature_extraction.extract_per_folder(
            args.output, tile_folders, model, conf, args.suffix, args.overwrite
        )


if __name__ == "__main__":
    main()
