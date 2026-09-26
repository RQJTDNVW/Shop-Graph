"""Locate ShopGraph's processed dataset in supported project layouts."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def processed_data_dir(required_file: str | None = None) -> Path:
    """Return the processed-data directory that contains ``required_file``.

    The downloadable project archive places the data under
    ``data/raw/processed`` while preprocessing scripts use ``data/processed``.
    Supporting both avoids an unnecessary multi-gigabyte data move.
    """
    candidates = (
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "data" / "raw" / "processed",
    )
    if required_file:
        for directory in candidates:
            if (directory / required_file).exists():
                return directory
    return next((directory for directory in candidates if directory.exists()), candidates[0])
