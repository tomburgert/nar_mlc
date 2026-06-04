import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.utils import add_mixed_noise
from data.transform import get_transforms
from data.dataset import (
    DeepGlobeDataset,
    UCMercedDataset,
    MimicCXRDataset,
    BirdsSierraNevDataset,
    Ben19V2Dataset,
    AIDMLDataset,
    BalanceMixDataset
)
from data.constants import BAND_99TH_PERCENTILES, BAND_NORM_STATS


from torchvision import transforms
from transform import Resize, RandomResizedCrop


class DataModule(pl.LightningDataModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def get_dataset(self, transform, split):
        if self.cfg.params.dataset == 'BigEarthNetV2':
            Dataset = Ben19V2Dataset
        elif self.cfg.params.dataset == 'DeepGlobe':
            Dataset = DeepGlobeDataset
        elif self.cfg.params.dataset == 'UCMerced':
            Dataset = UCMercedDataset
        elif self.cfg.params.dataset == 'BirdsSierraNev':
            Dataset = BirdsSierraNevDataset
        elif self.cfg.params.dataset == 'MIMICCXR':
            Dataset = MimicCXRDataset
        elif self.cfg.params.dataset == 'AID-ML':
            Dataset = AIDMLDataset
        if split == 'train':
            csv_path = self.cfg.dataset.train_csv
        if split == 'test':
            csv_path = self.cfg.dataset.test_csv
        return Dataset(self.cfg.dataset.lmdb_path, csv_path, self.cfg.dataset.labels_path, transform)

    def setup(self):

        self.trainset = self.get_dataset(
            transform=self.transform_tr,
            split='train'
        )

        # means = [0.4814, 0.4876, 0.4481]
        # stds  = [0.2173, 0.2013, 0.1950]
        # self.trainset_no_augment = self.get_dataset(
        #     transform=transforms.Compose([
        #         RandomResizedCrop(scale=(0.35, 1), resize_size=(256, 256), p=0.5),
        #         Resize((256, 256)),
        #         transforms.ToTensor(),
        #         transforms.Normalize(means, stds)]),
        #     split='train'
        # )
        self.trainset_no_augment = self.get_dataset(
            transform=self.transform_te,
            split='train'
        )
        if self.cfg.params.mixed_noise:
            self.trainset.labels, self.trainset.labels_gt = add_mixed_noise(
                self.trainset.labels,
                self.cfg.params.addn,
                self.cfg.params.addn,  # hack!!
                # self.cfg.params.subn
            )
        else:
            self.trainset.labels, self.trainset.labels_gt = add_mixed_noise(
                self.trainset.labels,
                self.cfg.params.addn,
                self.cfg.params.subn
            )

        if self.cfg.nrl.use_balancemix:
            self.trainset = BalanceMixDataset(self.trainset, alpha=self.cfg.nrl.bmix_alpha)
        
        self.testset  = self.get_dataset(
            transform=self.transform_te,
            split='test'
        )

        # self.trainset_with_augm_list = []
        # for i in range(len(self.transform_augm_list)):
        #     trainset_augm = copy.deepcopy(self.trainset)
        #     trainset_augm.transform = self.transform_augm_list[i]
        #     self.trainset_with_augm_list += [trainset_augm]

    def get_loader(self, dataset, shuffle):
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.params.batch_size,
            num_workers=self.cfg.params.num_workers,
            shuffle=shuffle,
            pin_memory=self.cfg.params.pin_memory
        )
        return dataloader

    # def get_loader_shuffle(self, dataset):
    #     dataloader = DataLoader(
    #         dataset,
    #         batch_size=self.cfg.dataset.batch_size,
    #         num_workers=self.cfg.dataset.num_workers,
    #         shuffle=True,
    #         pin_memory=self.cfg.dataset.pin_memory
    #     )
    #     return dataloader

    def train_dataloader(self, shuffle=True):
        return self.get_loader(self.trainset, shuffle)

    def train_dataloader_no_augment(self, shuffle=False):
        return self.get_loader(self.trainset_no_augment, shuffle)
    
    # def train_augm_dataloader(self):
    #     return list(map(lambda x: self.get_loader_shuffle(x), self.trainset_with_augm_list))

    def val_dataloader(self, shuffle=False):
        return self.get_loader(self.testset, shuffle)

    def test_dataloader(self, shuffle=False):
        return self.get_loader(self.testset, shuffle)


class BigEarthNetDataModule(DataModule):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.init_transforms()

    def init_transforms(self):
        self.train_country = os.path.basename(self.cfg.dataset.train_csv).split('_')[0].capitalize()
        self.test_country = list(map(lambda x: os.path.basename(x).split('_')[0].capitalize(), self.cfg.dataset.test_csv))
        if self.cfg.dataset.global_pctl:
            print('Global Pre-Normalization.')
            tr_percentiles = 10000
        elif self.cfg.dataset.all_percentiles and not self.cfg.dataset.global_pctl:
            print('All BEN Pre-Normalization.')
            tr_percentiles = list(BAND_99TH_PERCENTILES['All'].values())
        else:
            print('Country Pre-Normalization.')
            tr_percentiles = list(BAND_99TH_PERCENTILES[self.train_country].values())

        # currently train and test normalized by train norms (!)
        channel_global = 'Global' if self.cfg.dataset.global_pctl else 'Channel'
        if self.cfg.dataset.all_percentiles:
            print('All Percentiles', channel_global)
            means = list(BAND_NORM_STATS[channel_global]['All']['mean'].values())
            stds  = list(BAND_NORM_STATS[channel_global]['All']['std'].values())
        else:
            print('Country Percentiles', channel_global)
            means = list(BAND_NORM_STATS[channel_global][self.train_country]['mean'].values())
            stds  = list(BAND_NORM_STATS[channel_global][self.train_country]['std'].values())

        self.transform_tr = get_transforms(self.cfg.dataaug, self.cfg.dataaug.train_augmentation, means, stds, tr_percentiles, 'BigEarthNetV2')
        self.transform_te = get_transforms(self.cfg.dataaug, self.cfg.dataaug.test_augmentation, means, stds, tr_percentiles, 'BigEarthNetV2')


class DeepGlobeDataModule(DataModule):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.init_transforms()

    def init_transforms(self):
        means = [0.4095, 0.3808, 0.2836]
        stds  = [0.1509, 0.1187, 0.1081]
        self.transform_tr = get_transforms(self.cfg.dataaug, self.cfg.dataaug.train_augmentation, means, stds)
        self.transform_te = get_transforms(self.cfg.dataaug, self.cfg.dataaug.test_augmentation, means, stds)
        # self.transform_augm_list = list(map(lambda x: get_transforms(x, self.cfg.dataaug, means, stds), self.cfg.dataaug.augmentation_list))


class UCMercedDataModule(DataModule):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.init_transforms()

    def init_transforms(self):
        means = [0.4814, 0.4876, 0.4481]
        stds  = [0.2173, 0.2013, 0.1950]
        self.transform_tr = get_transforms(self.cfg.dataaug, self.cfg.dataaug.train_augmentation, means, stds, dataset='UCMerced')
        self.transform_te = get_transforms(self.cfg.dataaug, self.cfg.dataaug.test_augmentation, means, stds, dataset='UCMerced')
        # self.transform_augm_list = list(map(lambda x: get_transforms(x, self.cfg.dataaug, means, stds, dataset='UCMerced'), self.cfg.dataaug.augmentation_list))


class MimicCXRDataModule(DataModule):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.init_transforms()

    def init_transforms(self):
        means = [0.47307625]
        stds  = [0.30255994]
        self.transform_tr = get_transforms(self.cfg.dataaug, self.cfg.dataaug.train_augmentation, means, stds, dataset='MimicCXR')
        self.transform_te = get_transforms(self.cfg.dataaug, self.cfg.dataaug.test_augmentation, means, stds, dataset='MimicCXR')
        # self.transform_augm_list = list(map(lambda x: get_transforms(x, self.cfg.dataaug, means, stds, dataset='MimicCXR'), self.cfg.dataaug.augmentation_list))


class BirdsSierraNevDataModule(DataModule):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.init_transforms()

    def init_transforms(self):
        means = [0.3972]
        stds  = [0.1473]
        self.transform_tr = get_transforms(self.cfg.dataaug, self.cfg.dataaug.train_augmentation, means, stds, dataset='BirdsSierraNev')
        self.transform_te = get_transforms(self.cfg.dataaug, self.cfg.dataaug.test_augmentation, means, stds, dataset='BirdsSierraNev')
        # self.transform_augm_list = list(map(lambda x: get_transforms(x, self.cfg.dataaug, means, stds, dataset='BirdsSierraNev'), self.cfg.dataaug.augmentation_list))


class AIDMLDataModule(DataModule):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.init_transforms()

    def init_transforms(self):
        means = [0.4814, 0.4876, 0.4481]
        stds  = [0.2173, 0.2013, 0.1950]
        self.transform_tr = get_transforms(self.cfg.dataaug, self.cfg.dataaug.train_augmentation, means, stds, dataset='BirdsSierraNev')
        self.transform_te = get_transforms(self.cfg.dataaug, self.cfg.dataaug.test_augmentation, means, stds, dataset='BirdsSierraNev')


def get_datamodule(dataset):
    if dataset == 'BigEarthNetV2':
        return BigEarthNetDataModule
    elif dataset == 'DeepGlobe':
        return DeepGlobeDataModule
    elif dataset == 'UCMerced':
        return UCMercedDataModule
    elif dataset == 'MIMICCXR':
        return MimicCXRDataModule
    elif dataset == 'BirdsSierraNev':
        return BirdsSierraNevDataModule
    elif dataset == 'AID-ML':
        return AIDMLDataModule
    else:
        raise ValueError(f'{dataset} not implemented')
