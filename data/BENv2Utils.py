
"""
copied and slightly modified from https://git.tu-berlin.de/rsim/BENv2-DataLoading/-/blob/main/BENv2Utils.py
"""

from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

from safetensors.numpy import load as safetensor_load

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import lmdb


def stack_and_interpolate(
    bands: Dict[str, np.ndarray],
    order: Optional[Iterable[str]] = None,
    img_size: int = 120,
    upsample_mode: str = "nearest",
) -> np.array:
    """
    Supports 2D input (as values in the dict) with "nearest", "bilinear" and "bicubic" interpolation
    """

    def _interpolate(img_data):
        if not img_data.shape[-2:] == (img_size, img_size):
            return F.interpolate(
                torch.Tensor(np.float32(img_data)).unsqueeze(0).unsqueeze(0),
                (img_size, img_size),
                mode=upsample_mode,
                align_corners=True if upsample_mode in ["bilinear", "bicubic"] else None,
            ).squeeze().numpy()
        else:
            return np.float32(img_data)

    # if order is None, order is alphabetical
    if order is None:
        order = sorted(bands.keys())
    return np.stack([_interpolate(bands[x]) for x in order])


_s1_bandnames = ["VH", "VV"]
_s2_bandnames = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12", "B8A"]
_all_bandnames = _s2_bandnames + _s1_bandnames

STANDARD_BANDS = {
    "S1": _s1_bandnames,
    "S2": _s2_bandnames,
    "ALL": _all_bandnames,
    "RGB": ["B04", "B03", "B02"],
    "10m": ["B02", "B03", "B04", "B08"],
    "20m": ["B05", "B06", "B07", "B11", "B12", "B8A"],
    "60m": ["B01", "B09"],
    2: _s1_bandnames,
    10: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B11", "B12", "B8A"],
    12: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B11", "B12", "B8A", "VH", "VV"],
    3: ["B04", "B03", "B02"],
    4: ["B04", "B03", "B02", "B08"],
}


def resolve_band_combi(bands: Union[Iterable, str, int]) -> list:
    """
    Resolves a predefined combination of bands or a list of bands into a list of
    individual bands and checks if all bands contained are actual S1/S2 band names.

    :param bands: a combination of bands as defined in BAND_COMBINATION_PREDEFINTIONS
        or a list of bands, or a single band, e.g. "B02", ["B02", "B03"], 2, "S1", "S2"
    :return: a list of bands contained in the predefinition or the single band as list
    """
    if isinstance(bands, str):
        if bands in _s1_bandnames or bands in _s2_bandnames:
            bands = [bands]
        else:
            assert bands in STANDARD_BANDS.keys(), (
                "Band combination unknown, please use a list of strings or one of " f"{STANDARD_BANDS.keys()}"
            )
            bands = STANDARD_BANDS[bands]
    elif isinstance(bands, int):
        assert bands in STANDARD_BANDS.keys(), (
            "Band combination unknown, please use a list of strings or one of " f"{STANDARD_BANDS.keys()}"
        )
        bands = STANDARD_BANDS[bands]
    elif isinstance(bands, Iterable):
        for band in bands:
            assert band in _all_bandnames, f"Band '{band}' unknown"
    else:
        raise ValueError(f"Unknown type of bands: {type(bands)}")
    assert isinstance(bands, list), "Bands should be a list"
    return bands


class BENv2LDMBReader:
    def __init__(
        self,
        image_lmdb_file: Union[str, Path],
        metadata_file: Union[str, Path],
        metadata_snow_cloud_file: Optional[Union[str, Path]] = None,
        bands: Optional[Union[Iterable, str, int]] = None,
        process_bands_fn: Optional[Callable[[Dict[str, np.ndarray], List[str]], Any]] = None,
        process_labels_fn: Optional[Callable[[List[str]], Any]] = None,
    ):
        self.image_lmdb_file = image_lmdb_file
        self.env = None

        self.bands = bands if bands is not None else _all_bandnames
        self.bands = resolve_band_combi(self.bands)
        self.uses_s1 = any([x in _s1_bandnames for x in self.bands])
        self.uses_s2 = any([x in _s2_bandnames for x in self.bands])

        self.metadata = pd.read_parquet(metadata_file)
        if metadata_snow_cloud_file is not None:
            metadata_snow_cloud = pd.read_parquet(metadata_snow_cloud_file)
            self.metadata = pd.concat([self.metadata, metadata_snow_cloud])

        # self.lbls = {row["patch_id"]: row["labels"] for idx, row in self.metadata.iterrows()}
        self.lbls = {p: l for p, l in zip(self.metadata["patch_id"], self.metadata["labels"])}
        self.lbl_key_set = set(self.lbls.keys())
        # self.mapping = {row["patch_id"]: row["s1_name"] for idx, row in self.metadata.iterrows()}
        self.mapping = {p: s for p, s in zip(self.metadata["patch_id"], self.metadata["s1_name"])}

        # set mean and std based on bands selected
        self.mean = None
        self.std = None

        self.process_bands_fn = stack_and_interpolate
        self.process_labels_fn = process_labels_fn if process_labels_fn is not None else lambda x: x

        self._keys: Optional[set] = None
        self._S2_keys: Optional[set] = None
        self._S1_keys: Optional[set] = None

    def open_env(self):
        if self.env is None:
            self.env = lmdb.open(
                str(self.image_lmdb_file),
                readonly=True,
                lock=False,
                meminit=False,
                readahead=True,
                map_size=8 * 1024**3,  # 8GB blocked for caching
                max_spare_txns=16,  # expected number of concurrent transactions (e.g. threads/workers)
            )

    def keys(self, update: bool = False):
        self.open_env()
        if self._keys is None or update:
            assert self.env is not None, "Environment not opened yet"
            with self.env.begin() as txn:
                self._keys = set(txn.cursor().iternext(values=False))
            self._keys = {x.decode() for x in self._keys}
        return self._keys

    def S2_keys(self, update: bool = False):
        if self._S2_keys is None or update:
            self._S2_keys = {key for key in self.keys(update) if key.startswith("S2")}
        return self._S2_keys

    def S1_keys(self, update: bool = False):
        if self._S1_keys is None or update:
            self._S1_keys = {key for key in self.keys(update) if key.startswith("S1")}
        return self._S1_keys

    def __getitem__(self, key: str):
        # the key is the name of the S2v2 patch

        # open lmdb file if not opened yet
        self.open_env()
        img_data_dict: dict = {}
        if self.uses_s2:
            assert self.env is not None, "Environment not opened yet"
            # read image data for S2v2
            with self.env.begin(write=False, buffers=True) as txn:
                byte_data = txn.get(key.encode())
            img_data_dict.update(safetensor_load(bytes(byte_data)))

        if self.uses_s1:
            # read image data for S1
            assert self.mapping is not None, "S1 bands are used, but no mapping is provided"
            s1_key = self.mapping[key]
            assert self.env is not None, "Environment not opened yet"
            with self.env.begin(write=False, buffers=True) as txn:
                byte_data = txn.get(s1_key.encode())
            img_data_dict.update(safetensor_load(bytes(byte_data)))

        assert isinstance(self.bands, list), "Bands should be a list"
        img_data_dict = {k: v for k, v in img_data_dict.items() if k in self.bands}

        img_data = self.process_bands_fn(img_data_dict, self.bands)
        labels = self.lbls[key] if key in self.lbl_key_set else []
        labels = self.process_labels_fn(labels)

        return img_data, labels
