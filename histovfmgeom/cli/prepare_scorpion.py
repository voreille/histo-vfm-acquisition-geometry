"""Unified SCORPION dataset preparation CLI.

Pipeline:
  1. (optional) Download SCORPION_dataset.zip from Zenodo and unzip.
  2. Prepare: copy/rename from the nested raw structure to a flat intermediate dir.
  3. Tile: rescale to target MPP, pre-crop, and cut into tiles.
  4. Write metadata CSV.

Raw structure (from Zenodo zip):
  <raw_dir>/slide_N/sample_N/{AT2,DP200,GT450,P1000,Philips}.jpg

Tile output naming:
  <tile_dir>/{slide_id}-{sample_id}-tile_{i}_{j}-{scanner_id}.jpg
"""

from __future__ import annotations

import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path

import click
import pandas as pd
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

_ZENODO_URL = "https://zenodo.org/records/16517924/files/SCORPION_dataset.zip"
_DEFAULT_RAW_DIR = Path("data/raw/SCORPION_dataset")
_DEFAULT_TILE_DIR = Path("data/processed/SCORPION_tiles_224px_0p5mpp")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _center_crop(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    w, h = image.size
    left = (w - target_size[0]) // 2
    top = (h - target_size[1]) // 2
    return image.crop((left, top, left + target_size[0], top + target_size[1]))


def _download_with_progress(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _reporthook(count: int, block_size: int, total_size: int) -> None:
        if not hasattr(_reporthook, "bar"):
            _reporthook.bar = tqdm(  # type: ignore[attr-defined]
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=dest.name,
            )
        _reporthook.bar.update(block_size)  # type: ignore[attr-defined]

    urllib.request.urlretrieve(url, dest, reporthook=_reporthook)
    if hasattr(_reporthook, "bar"):
        _reporthook.bar.close()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def download_and_extract(raw_dir: Path) -> None:
    """Download SCORPION_dataset.zip from Zenodo and extract next to raw_dir."""
    zip_path = raw_dir.parent / "SCORPION_dataset.zip"

    click.echo(f"Downloading {_ZENODO_URL} → {zip_path}")
    _download_with_progress(_ZENODO_URL, zip_path)

    extract_root = raw_dir.parent
    click.echo(f"Extracting {zip_path} → {extract_root}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)

    zip_path.unlink()
    click.echo("Removed zip archive.")


def tile_raw_dataset(
    raw_dir: Path,
    tile_dir: Path,
    tile_size: int,
    input_mpp: float,
    target_mpp: float,
    rescale: bool,
    precrop: bool,
) -> None:
    """Read raw SCORPION images, rescale, precrop, tile, and write to tile_dir."""
    tile_dir.mkdir(parents=True, exist_ok=True)

    # raw layout: slide_N / sample_N / {scanner}.jpg
    jpg_files = sorted(raw_dir.rglob("*.jpg"))
    if not jpg_files:
        raise FileNotFoundError(f"No .jpg files found under {raw_dir}")

    scale_factor = input_mpp / target_mpp

    for jpg in tqdm(jpg_files, desc="Tiling"):
        scanner_id = jpg.stem                 # e.g. AT2
        sample_id = jpg.parent.name           # e.g. sample_1
        slide_id = jpg.parent.parent.name     # e.g. slide_1

        image = Image.open(jpg).convert("RGB")

        if rescale:
            new_size = (
                int(image.width * scale_factor),
                int(image.height * scale_factor),
            )
            image = image.resize(new_size, resample=Image.LANCZOS)

        if precrop:
            tw = (image.width // tile_size) * tile_size
            th = (image.height // tile_size) * tile_size
            image = _center_crop(image, (tw, th))

        n_x = image.width // tile_size
        n_y = image.height // tile_size

        for i in range(n_x):
            for j in range(n_y):
                tile = image.crop((
                    i * tile_size,
                    j * tile_size,
                    (i + 1) * tile_size,
                    (j + 1) * tile_size,
                ))
                name = f"{slide_id}-{sample_id}-tile_{i}_{j}-{scanner_id}.jpg"
                tile.save(tile_dir / name)


def write_metadata(tile_dir: Path) -> Path:
    """Scan tile_dir for tiles and write metadata.csv."""
    records = []
    for jpg in sorted(tile_dir.glob("*.jpg")):
        stem = jpg.stem  # slide_id-sample_id-tile_i_j-scanner_id
        parts = stem.split("-")
        # parts: [slide_id, sample_id, "tile_i_j", scanner_id]
        if len(parts) < 4:
            logger.warning("Unexpected filename format, skipping: %s", jpg.name)
            continue
        slide_id = parts[0]
        sample_id = parts[1]
        tile_local = parts[2]          # tile_i_j
        scanner_id = parts[3]
        tile_id = f"{slide_id}-{sample_id}-{tile_local}"
        records.append({
            "slide_id": slide_id,
            "sample_id": sample_id,
            "scanner_id": scanner_id,
            "tile_id": tile_id,
            "filename": stem,
        })

    df = pd.DataFrame(records)
    csv_path = tile_dir / "metadata.csv"
    df.to_csv(csv_path, index=False)
    click.echo(f"Wrote metadata ({len(df)} rows) → {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--download",
    is_flag=True,
    default=False,
    help="Download SCORPION_dataset.zip from Zenodo before processing.",
)
@click.option(
    "--raw-dir",
    type=click.Path(path_type=Path),
    default=_DEFAULT_RAW_DIR,
    show_default=True,
    help="Path to the unzipped SCORPION raw dataset (slide_N/sample_N structure).",
)
@click.option(
    "--tile-dir",
    type=click.Path(path_type=Path),
    default=_DEFAULT_TILE_DIR,
    show_default=True,
    help="Output directory for tiles and metadata.csv.",
)
@click.option(
    "--tile-size",
    type=int,
    default=224,
    show_default=True,
    help="Tile side length in pixels.",
)
@click.option(
    "--input-mpp",
    type=float,
    default=0.78125,
    show_default=True,
    help="Microns-per-pixel of the raw images.",
)
@click.option(
    "--target-mpp",
    type=float,
    default=0.5,
    show_default=True,
    help="Microns-per-pixel of the output tiles.",
)
@click.option(
    "--no-rescale",
    is_flag=True,
    default=False,
    help="Skip MPP rescaling.",
)
@click.option(
    "--no-precrop",
    is_flag=True,
    default=False,
    help="Skip pre-cropping to a tile-aligned size.",
)
def main(
    download: bool,
    raw_dir: Path,
    tile_dir: Path,
    tile_size: int,
    input_mpp: float,
    target_mpp: float,
    no_rescale: bool,
    no_precrop: bool,
) -> None:
    """Prepare the SCORPION dataset: download, tile, and write metadata."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if download:
        download_and_extract(raw_dir)

    if not raw_dir.exists():
        raise click.ClickException(
            f"Raw dataset directory not found: {raw_dir}\n"
            "Run with --download to fetch it from Zenodo, or pass --raw-dir."
        )

    click.echo(f"Tiling {raw_dir} → {tile_dir}")
    tile_raw_dataset(
        raw_dir=raw_dir,
        tile_dir=tile_dir,
        tile_size=tile_size,
        input_mpp=input_mpp,
        target_mpp=target_mpp,
        rescale=not no_rescale,
        precrop=not no_precrop,
    )

    write_metadata(tile_dir)
    click.echo("Done.")


if __name__ == "__main__":
    main()
