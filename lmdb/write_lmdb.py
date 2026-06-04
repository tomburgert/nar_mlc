import rasterio
from pathlib import Path
from typing import List

from tqdm import tqdm
import lmdb
import numpy as np


def read_tif(file_path, read_channels=None):
    """
    Path to `tif` file.

    If read_channel is `None` all channels/bands are read from the file.
    Otherwise the value of read_channel is used.
    """
    # https://gitlab.tubit.tu-berlin.de/rsim/bigearthnet-models-tf/blob/master/BigEarthNet.py
    # Needs to convert bands to
    with rasterio.open(file_path) as tif_image:
        num_channels = tif_image.count
        channels = range(1, num_channels + 1)
        read_channels = channels if read_channels is None else read_channels
        data = tif_image.read(read_channels)
    return data


def _write_lmdb(
    patch_paths: List[Path],
    lmdb_path: Path = Path("lmdb.db"),
    lmdb_max_size: int = 2**40
) -> None:
    """
    The function writes an LMDB archive.
    A list of patch paths is processed individually.
    The output archive is written to `lmdb_path`.

    `lmdb_max_size` is the _theoretical_ upper size for the LMDB archive.
    Usually, there should be no need to change this default, as 1TiB (the default)
    is big enough for most encodings and shouldn't even bother if not as much disk-space
    is available. Though some users have reported that they had to change this value
    to get it working which I couldn't reproduce.
    """
    if len(patch_paths) == 0:
        raise ValueError(
            "No patches were provided! Maybe provided wrong Sentinel directory?"
        )
    env = lmdb.open(str(lmdb_path), map_size=lmdb_max_size, readonly=False)
    
    with env.begin(write=True) as txn:
        for patch_path in tqdm(patch_paths):
            patch_name = patch_path.stem
            tifs = read_tif(patch_path)
            tifs = np.moveaxis(tifs, 0, -1)
            txn.put(patch_name.encode(), tifs.dumps())

    env.close()


# write aerial lmdb
aerial_paths = list(Path('/workspace/datasets/TreeSatAI/raw').glob('*.tif'))
_write_lmdb(aerial_paths, Path('/workspace/datasets/TreeSatAI/aerial_lmdb.db'))

s2_60m_paths = list(Path('/workspace/datasets/TreeSatAI/s2/60m').glob('*.tif'))
_write_lmdb(s2_60m_paths, Path('/workspace/datasets/TreeSatAI/s2_60m_lmdb.db'))
