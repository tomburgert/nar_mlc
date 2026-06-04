import pickle

import lmdb
import numpy as np
import pandas as pd
import torch
from safetensors.numpy import load

from datasets import load_dataset
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset

from constants import (
    BEN19_NAME2IDX,
    DEEPGLOBE_NAME2IDX,
    UCMERCED_NAME2IDX,
    MIMICCXR_NAME2IDX,
    BIRDSSIERRANEV_NAME2IDX,
    AID_IDX2NAME,
)
from data.BENv2Utils import BENv2LDMBReader

from transform import Resize


class BaseDataset(Dataset):
    def __init__(self, lmdb_path, csv_path, labels_path, transform=None):
        """
        Parameter
        ---------
        lmdb_path      : path to the LMDB file for efficiently loading the patches.
        csv_path       : path to a csv file containing the patch names that will make up this split
        transform_mode:  specifies the image transform mode which determines the augmentations
                         to be applied to the image
        """
        self.env = None 
        self.lmdb_path = lmdb_path
        self.patch_names = self.read_csv(csv_path)
        print(f'load split from {csv_path} with {len(self.patch_names)} patches')

        self.transform = transform

    def read_csv(self, csv_data):
        return pd.read_csv(csv_data, header=None).to_numpy()[:, 0]

    def read_labels(self, meta_data_path, patch_names):
        df = pd.read_parquet(meta_data_path)
        df_subset = df.set_index('name').loc[self.patch_names].reset_index(inplace=False)
        
        string_labels = df_subset['labels'].tolist()
        multihot_labels = np.array(list(map(self.convert_to_multihot, string_labels)))
        return multihot_labels
    
    def convert_to_multihot(self, labels):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        """Get item at position idx of Dataset."""
        if self.env is None:
            self.env = lmdb.open(str(self.lmdb_path), readonly=True, lock=False, meminit=False, readahead=True)

        patch_name = self.patch_names[idx]
            
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(patch_name.encode('utf-8'))
            patch = pickle.loads(byteflow)

        label = self.labels[idx]
        weight = self.weights[idx]
        patch = self.transform(patch) if self.transform is not None else patch

        return patch, label, weight, idx

    def __len__(self):
        """Get length of Dataset."""
        return len(self.patch_names)


class Ben19V2Dataset(BaseDataset):
    def __init__(self, lmdb_path, csv_path, labels_path, transform=None, active_classes=None,
                 rgb_only=False, discard_empty_labels=True):
        super().__init__(lmdb_path, csv_path, labels_path, transform)
        """
        Parameter
        ---------
        lmdb_path      : path to the LMDB file for efficiently loading the patches.
        csv_path       : path to a csv file containing the patch names that will make up this split
        transform_mode:  specifies the image transform mode which determines the augmentations
                         to be applied to the image
        """  
        self.band_ordering = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
        self.return_patchname = False
        self.reader = BENv2LDMBReader(
            image_lmdb_file=lmdb_path,
            metadata_file=labels_path,
            bands=self.band_ordering,
            process_bands_fn=None,
            process_labels_fn=self.convert_to_multihot,
        )

        self.create_df_labels()
        self.labels = self.read_labels(self.df_labels)
        assert (self.labels.sum(axis=1) == 0).sum() == 0
        assert (self.labels.sum(axis=0) == 0).sum() == 0

        self.labels_gt = self.labels.copy()                                     # numpy
        self.weights = torch.ones(self.labels.shape)                            # tensor

    def create_df_labels(self):
        self.df_labels = pd.DataFrame(list(self.reader.lbls.items()), columns=['name', 'labels'])
        self.df_labels = self.df_labels[self.df_labels['name'].isin(self.patch_names)]
        self.df_labels['name'] = pd.Categorical(self.df_labels['name'], categories=self.patch_names, ordered=True)  # sort dataframe by self.patch_names
        self.df_labels = self.df_labels.sort_values(by='name').reset_index()

    def read_labels(self, lbl_parquet):
        string_labels = lbl_parquet['labels'].tolist()
        multihot_labels = np.array(list(map(self.convert_to_multihot, string_labels)))
        return multihot_labels

    def convert_to_multihot(self, labels):
        multihot = np.zeros(19)
        indices = [BEN19_NAME2IDX[label] for label in labels]
        multihot[indices] = 1
        return multihot
    
    def __getitem__(self, idx):
        patch_name = self.patch_names[idx]
        patch, label_original = self.reader[patch_name]
        patch = np.moveaxis(patch, 0, -1)  # s.t. ToTensor can reverse to beginning
        patch = self.transform(patch) if self.transform is not None else patch

        label = self.labels[idx]
        weight = self.weights[idx]

        return patch, label, weight, idx


class DeepGlobeDataset(BaseDataset):
    def __init__(self, lmdb_path, csv_path, labels_path, transform=None):
        super().__init__(lmdb_path, csv_path, labels_path, transform)

        self.labels = self.read_labels(labels_path, self.patch_names)           # numpy 
        self.labels_gt = self.labels.copy()                                     # numpy
        self.weights = torch.ones(self.labels.shape)                            # tensor
    
    def convert_to_multihot(self, labels):
        multihot = np.zeros(6)
        indices = [DEEPGLOBE_NAME2IDX[label] for label in labels]
        multihot[indices] = 1
        return multihot


class UCMercedDataset(BaseDataset):
    def __init__(self, lmdb_path, csv_path, labels_path, transform=None):
        super().__init__(lmdb_path, csv_path, labels_path, transform)

        self.labels = self.read_labels(labels_path, self.patch_names)           # numpy 
        self.labels_gt = self.labels.copy()                                     # numpy
        self.weights = torch.ones(self.labels.shape)                            # tensor
    
    def convert_to_multihot(self, labels):
        multihot = np.zeros(len(UCMERCED_NAME2IDX.keys()))
        indices = [UCMERCED_NAME2IDX[label] for label in labels]
        multihot[indices] = 1
        return multihot

    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(str(self.lmdb_path), readonly=True, lock=False, meminit=False, readahead=True)

        patch_name = self.patch_names[idx]

        with self.env.begin(write=False) as txn:
            safetensor_dict = load(txn.get(patch_name.encode()))

        patch = np.stack([safetensor_dict[key] for key in ["Red", "Green", "Blue"]], axis=2)

        label = self.labels[idx]
        weight = self.weights[idx]
        patch = self.transform(patch) if self.transform is not None else patch

        return patch, label, weight, idx
    

class MimicCXRDataset(BaseDataset):
    def __init__(self, lmdb_path, csv_path, labels_path, transform=None):
        super().__init__(lmdb_path, csv_path, labels_path, transform)

        self.labels = self.read_labels(labels_path, self.patch_names)           # numpy 
        self.labels_gt = self.labels.copy()                                     # numpy
        self.weights = torch.ones(self.labels.shape)                            # tensor
    
    def convert_to_multihot(self, labels):
        multihot = np.zeros(len(MIMICCXR_NAME2IDX.keys()))
        indices = [MIMICCXR_NAME2IDX[label] for label in labels]
        multihot[indices] = 1
        return multihot

    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(str(self.lmdb_path), readonly=True, lock=False, meminit=False, readahead=True)
                
        patch_name = self.patch_names[idx]

        with self.env.begin(write=False) as txn:
            byteflow = txn.get(patch_name.encode('utf-8'))
            try:
                patch = pickle.loads(byteflow)
            except:  # noqa: E722
                raise ValueError(f'patch loading failed for patch {patch_name}')
            patch = patch.astype(np.uint8)
        
        patch = patch[:, :, None]
                
        label = self.labels[idx]
        weight = self.weights[idx]
        patch = self.transform(patch) if self.transform is not None else patch

        return patch, label, weight, idx


class BirdsSierraNevDataset(BaseDataset):
    def __init__(self, lmdb_path, csv_path, labels_path, transform=None):
        super().__init__(lmdb_path, csv_path, labels_path, transform)

        self.labels = self.read_labels(labels_path, self.patch_names)           # numpy 
        self.labels_gt = self.labels.copy()                                     # numpy
        self.weights = torch.ones(self.labels.shape)                            # tensor
    
    def convert_to_multihot(self, labels):
        multihot = np.zeros(len(BIRDSSIERRANEV_NAME2IDX.keys()))
        indices = [BIRDSSIERRANEV_NAME2IDX[label] for label in labels]
        multihot[indices] = 1
        return multihot

    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(str(self.lmdb_path), readonly=True, lock=False, meminit=False, readahead=True)
                
        patch_name = self.patch_names[idx]

        with self.env.begin(write=False) as txn:
            byteflow = txn.get(patch_name.encode('utf-8'))
            try:
                patch = pickle.loads(byteflow)
            except:  # noqa: E722
                raise ValueError(f'patch loading failed for patch {patch_name}')
            patch = patch.astype(np.uint8)
        
        patch = patch[:, :, None]
                
        label = self.labels[idx]
        weight = self.weights[idx]
        patch = self.transform(patch) if self.transform is not None else patch

        return patch, label, weight, idx


class AIDMLDataset(Dataset):
    def __init__(self, lmdb_path=None, csv_path="train", labels_path=None, transform=None):
        assert csv_path in ["train", "val", "test"], "csv_path must be 'train', 'val', or 'test'"

        full_dataset = load_dataset("jonathan-roberts1/AID_MultiLabel")['train']
        self.transform = transform
        self.resize = Resize((256, 256))
        self.csv_path = csv_path

        self.all_images = full_dataset['image']
        self.all_labels = np.array([self.convert_to_multihot(lbls) for lbls in full_dataset['label']])

        self.split_indices = self.get_split_indices(seed=2024)

        self.labels = self.all_labels[self.split_indices]
        self.weights = torch.ones((len(self.labels), len(AID_IDX2NAME)))

    def convert_to_multihot(self, label_list):
        multihot = np.zeros(len(AID_IDX2NAME), dtype=np.float32)
        for label in label_list:
            multihot[label] = 1
        return multihot

    def get_split_indices(self, seed=2024):
        all_idx = np.arange(len(self.all_labels))
        train_val_idx, test_idx = train_test_split(
            all_idx, test_size=0.15, random_state=seed, shuffle=True
        )
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=0.1765, random_state=seed, shuffle=True
        )

        if self.csv_path == "train":
            return train_idx
        elif self.csv_path == "val":
            return val_idx
        else:  # "test"
            return test_idx

    def __getitem__(self, idx):
        real_idx = self.split_indices[idx]
        image = np.array(self.all_images[real_idx].convert("RGB"))
        image = self.resize(image)

        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[idx]
        weight = self.weights[idx]

        return image, label, weight, idx

    def __len__(self):
        return len(self.split_indices)


class MinoritySampler:
    """
    Oversamples examples containing rare labels.
    For each example i with multi-hot y_i, we compute
      w_i = sum_k [ y_{i,k} / freq_k ]
    and normalize into a probability distribution.
    """
    def __init__(self, labels: np.ndarray, eps: float = 1e-6):
        # labels: (N, K) multi-hot numpy array
        # compute class frequencies
        freq = labels.sum(axis=0) / labels.shape[0]
        inv_freq = 1.0 / (freq + eps)
        # sample weight per example: sum of inv-freqs of its positive classes
        weights = (labels * inv_freq[None]).sum(axis=1)
        self.probs = weights / weights.sum()

    def sample_idx(self) -> int:
        return int(np.random.choice(len(self.probs), p=self.probs))


class UniformSampler:
    """Uniform random sampler over dataset indices."""
    def __init__(self, dataset_size: int):
        self.N = dataset_size

    def sample_idx(self) -> int:
        return np.random.randint(0, self.N)


class BalanceMixDataset(Dataset):
    """
    For each item returns a mixup of
      (x1,y1,w1,idx1) drawn via minority sampler
      (x2,y2,w2,idx2) drawn uniformly,
    mixed with lambda ~ Beta(alpha, alpha), enforced > 0.5.
    """
    def __init__(
        self,
        base_dataset: Dataset,
        alpha: float = 0.4,
    ):
        self.base = base_dataset
        self.labels      = base_dataset.labels
        self.weights     = base_dataset.weights
        self.patch_names = getattr(base_dataset, 'patch_names', None)
        self.labels_gt   = getattr(base_dataset, 'labels_gt', None)
        # load labels from underlying dataset (assumes attribute .labels as numpy array)
        assert self.labels is not None, "Base dataset must expose .labels (np.ndarray)"
        # samplers
        self.minority_sampler = MinoritySampler(self.labels)
        self.uniform_sampler  = UniformSampler(len(self.base))
        # mixup parameter
        self.alpha = alpha
        # Beta distribution for mixup
        self.beta_dist = torch.distributions.Beta(alpha, alpha)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        # draw two indices
        i_rand = self.uniform_sampler.sample_idx()
        # i_min = self.uniform_sampler.sample_idx()
        i_min = self.minority_sampler.sample_idx()

        x1, y1, w1, _ = self.base[i_rand]
        x2, y2, w2, _ = self.base[i_min]

        # sample lambda and enforce >= 0.5
        lam = float(self.beta_dist.sample().item())
        lam = max(lam, 1.0 - lam)

        # mix inputs
        x_mix = lam * x1 + (1.0 - lam) * x2
        # mix labels and weights
        y_mix = lam * y1 + (1.0 - lam) * y2
        w_mix = lam * w1 + (1.0 - lam) * w2

        # for logging we can return primary idx; metrics still aggregate over batch idxs
        return x_mix, y_mix, w_mix, i_rand
