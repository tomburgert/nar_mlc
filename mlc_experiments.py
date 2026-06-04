import sys
import yaml
import os

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from config import NoiseMLCConfig


print(hydra)

cs = ConfigStore.instance()
cs.store(name="base_config", node=NoiseMLCConfig)


def update_dataset_parameter(cfg, dataset):

    with open('conf/datasets.yaml', "r") as f:
        yaml_file = yaml.safe_load(f)

    cfg.dataset.lmdb_path    = yaml_file[dataset]['lmdb_path']
    cfg.dataset.labels_path  = yaml_file[dataset]['labels_path']
    cfg.dataset.train_csv    = yaml_file[dataset]['train_csv']
    cfg.dataset.val_csv      = yaml_file[dataset]['val_csv']
    cfg.dataset.test_csv     = yaml_file[dataset]['test_csv']
    cfg.dataset.num_classes  = yaml_file[dataset]['num_classes']
    cfg.dataset.num_channels = yaml_file[dataset]['num_channels']

    return cfg


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    if cfg.params.slurm_bypass:
        print('Bypassing slurm, using GPU: {}'.format(cfg.params.cuda_no))
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.params.cuda_no
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    print(OmegaConf.to_yaml(cfg))

    import torch

    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.loggers import CSVLogger
    from pytorch_lightning.callbacks import ModelCheckpoint

    from base import BaseModel
    from models import get_network

    sys.path.append('data')
    from data.datamodule import get_datamodule  # noqa: E402

    seed_everything(cfg.params.seed, workers=True)
    torch.set_float32_matmul_precision('high')
    cfg = update_dataset_parameter(cfg, cfg.params.dataset)

    DataModule = get_datamodule(cfg.params.dataset)
    print(DataModule)
    dm = DataModule(cfg)
    dm.setup()

    callbacks = []
    if cfg.logging.save_checkpoint:
        callbacks += [ModelCheckpoint(monitor='val_APmac', filename='best_model_val0', mode='max')]

    network = get_network(cfg.model.name, cfg.model.pretrained, cfg.dataset.num_channels, cfg.dataset.num_classes)
    model = BaseModel(cfg, dm, network)

    trainer = Trainer(
        accelerator='gpu',
        callbacks=callbacks,
        devices=[0],
        enable_checkpointing=cfg.logging.save_checkpoint,
        max_epochs=cfg.params.max_epochs,
        logger=CSVLogger(save_dir=cfg.logging.exp_dir),
        deterministic=True
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
