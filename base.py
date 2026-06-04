import os
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
import torch.nn.functional as F

from torchmetrics import MetricCollection
from torchmetrics.classification import F1Score, AveragePrecision

import pytorch_lightning as pl

from utils import LinearWarmupCosineAnnealingLR
from loss import ELR, SelfAdaptiveTrainingCE, AsymmetricLossOptimized, RALossOptimized
from utils import EMA
from models import get_network


class BaseModel(pl.LightningModule):
    def __init__(self, cfg, datamodule, network):
        super().__init__()
        self.cfg = cfg

        self.model = network

        # Second model for co-teaching if enabled
        if self.cfg.nrl.co_teaching:
            self.model2 = get_network(cfg.model.name, cfg.model.pretrained, cfg.dataset.num_channels, cfg.dataset.num_classes)

        # if self.cfg.nrl.use_ema:
        #     self.ema = EMA(
        #         decay=self.cfg.nrl.ema_decay,
        #         mode=self.cfg.nrl.ema_mode,
        #         warmup_epoch=self.cfg.nrl.ema_warmup_epoch
        #     )
        self.save_hyperparameters('cfg')
        
        self.datamodule = datamodule
        self.init_criterion()
        self.criterion = self.criterion_wrapper

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.training_no_augment_step_outputs = []

        metrics = self.init_metrics(self.cfg.dataset.num_classes)
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        
        self.probs_train = None
        self.probs_train_ema = None

        if self.cfg.nrl.co_teaching:
            self.training_step_outputs2 = []
            self.probs_train1 = None
            self.weights1 = torch.ones_like(torch.tensor(self.datamodule.trainset.labels, dtype=torch.float32))
            self.weights2 = torch.ones_like(torch.tensor(self.datamodule.trainset.labels, dtype=torch.float32))
        
    def forward(self, x):
        if self.cfg.nrl.co_teaching:
            return self.model(x), self.model2(x)
        return self.model(x)

    def init_criterion(self):
        if self.cfg.nrl.use_elr:
            self.elr_loss = ELR(
                N=len(self.datamodule.trainset),
                num_classes=self.cfg.dataset.num_classes,
                lam=self.cfg.nrl.elr_lam,
                beta=self.cfg.nrl.elr_beta,
            )
        if self.cfg.nrl.use_sat:
            self.sat_loss = SelfAdaptiveTrainingCE(
                labels=self.datamodule.trainset.labels.copy(),
                num_classes=self.cfg.dataset.num_classes,
                momentum=self.cfg.nrl.sat_momentum,
                warmup_epochs=self.cfg.nrl.sat_warmup_epochs,
            )
        if self.cfg.nrl.use_asl:
            self.asl_loss = AsymmetricLossOptimized(
                gamma_neg=self.cfg.nrl.asl_gamma_neg,
                gamma_pos=self.cfg.nrl.asl_gamma_pos,
                clip=self.cfg.nrl.asl_clip,
            )
        if self.cfg.nrl.use_ral:
            self.ral_loss = RALossOptimized(
                gamma_neg=self.cfg.nrl.asl_gamma_neg,
                gamma_pos=self.cfg.nrl.asl_gamma_pos,
                clip=self.cfg.nrl.asl_clip,
                lamb=self.cfg.nrl.ral_lamb,
                epsilon_neg=self.cfg.nrl.ral_epsilon_neg,
                epsilon_pos=self.cfg.nrl.ral_epsilon_pos,
                epsilon_pos_pow=self.cfg.nrl.ral_epsilon_pos_pow,
            )

    def criterion_wrapper(self, logits, y, weight, idx):
        if self.cfg.nrl.use_nrl and not self.cfg.nrl.use_elr and not self.cfg.nrl.use_sat:
            return F.binary_cross_entropy_with_logits(logits, y, weight=weight)

        # ELR
        elif not self.cfg.nrl.use_nrl and self.cfg.nrl.use_elr:
            return self.elr_loss(logits, y, idx)
        elif self.cfg.nrl.use_nrl and self.cfg.nrl.use_elr:
            return self.elr_loss(logits, y, idx, weight)

        # SAT
        elif not self.cfg.nrl.use_nrl and self.cfg.nrl.use_sat:
            return self.sat_loss(logits, y, idx, self.current_epoch)
        elif self.cfg.nrl.use_nrl and self.cfg.nrl.use_sat:
            return self.sat_loss(logits, y, idx, self.current_epoch, weight=weight)

        # ASL
        elif not self.cfg.nrl.use_nrl and self.cfg.nrl.use_asl:
            return self.asl_loss(logits, y)

        # RAL
        elif not self.cfg.nrl.use_nrl and self.cfg.nrl.use_ral:
            return self.ral_loss(logits, y)
        else:
            # fallback to plain BCE
            return F.binary_cross_entropy_with_logits(logits, y, weight=weight)

    def on_train_start(self):
        
        # self.log_noisy_labels()
            
        self.y_start = self.datamodule.trainset.labels.copy()
        
        if self.cfg.tracking.should_track_labels:
            self.track_labels(self.y_start, col_name='y_start')
            self.track_labels(self.datamodule.trainset.labels_gt, col_name='y_gt')

        if self.cfg.nrl.co_teaching:
            self.weights1 = self.weights1.to(self.device)
            self.weights2 = self.weights2.to(self.device)

    def on_train_epoch_start(self):
        if self.cfg.tracking.should_track_labels:
            self.track_labels(self.datamodule.trainset.labels)
        if self.cfg.tracking.should_track_weights:
            self.track_weights(self.datamodule.trainset.weights)
        if self.cfg.tracking.should_track_noise_statistics:
            # self.log_noise_statistics(self.get_noise_statistics(
            #     self.y_start,
            #     self.datamodule.trainset.labels_gt,
            #     self.datamodule.trainset.labels,
            #     self.datamodule.trainset.weights
            # ))
            self.log_subtractive_flip_statistics(self.get_subtractive_flip_statistics(
                self.y_start,
                self.datamodule.trainset.labels_gt,
                self.datamodule.trainset.labels
            ))

    def update_probabilities(self):
        if self.cfg.nrl.use_nrl:
            if self.cfg.nrl.use_ema and self.current_epoch >= self.cfg.nrl.ema_warmup_epoch:
                self.probs_train_ema.mul_(self.cfg.nrl.ema_decay).add_(self.probs_train, alpha=1 - self.cfg.nrl.ema_decay)
                self.probs_train = self.probs_train_ema.clone()
            if self.cfg.nrl.co_teaching:
                _, self.weights2 = self.NRL_thresh(self.y_start, self.probs_train)
                _, self.weights1 = self.NRL_thresh(self.y_start, self.probs_train2)
            else:
                if self.cfg.nrl.use_oracle:
                    self.datamodule.trainset.labels, self.datamodule.trainset.weights = self.NRL_oracle(self.datamodule.trainset.labels_gt, self.y_start, self.probs_train)
                else:
                    self.datamodule.trainset.labels, self.datamodule.trainset.weights = self.NRL_thresh(self.y_start, self.probs_train)

    def training_step(self, batch, batch_idx):
        x, y, weight, idx = batch

        if self.cfg.nrl.co_teaching:
            logits1 = self.model(x)
            logits2 = self.model2(x)

            weights2_idx = self.weights2[idx]
            weights1_idx = self.weights1[idx]
            loss1 = self.criterion(logits1, y, weights2_idx, idx)
            loss2 = self.criterion(logits2, y, weights1_idx, idx)

            # Compute probabilities
            probs1 = torch.sigmoid(logits1)
            probs2 = torch.sigmoid(logits2)

            output1 = dict(y=y, idx=idx, loss=loss1, probs=probs1)
            output2 = dict(y=y, idx=idx, loss=loss2, probs=probs2)
            self.training_step_outputs.append(output1)
            self.training_step_outputs2.append(output2)
            return output1
        else:
            logits    = self.forward(x)
            loss      = self.criterion(logits, y, weight, idx)
            probs     = torch.sigmoid(logits)
            output    = dict(y=y, idx=idx, loss=loss, probs=probs)
            self.training_step_outputs.append(output)
            return output

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, weight, idx = batch

        if self.cfg.nrl.co_teaching:
            logits1 = self.model(x)
            logits2 = self.model2(x)

            loss1 = self.criterion(logits1, y, weight, idx)
            loss2 = self.criterion(logits2, y, weight, idx)
            loss = (loss1 + loss2) / 2

            # Compute probabilities
            probs1 = torch.sigmoid(logits1)
            probs2 = torch.sigmoid(logits2)
            probs = (probs1 + probs2) / 2
        else:
            logits    = self.forward(x)
            loss      = self.criterion(logits, y, weight, idx)
            probs     = torch.sigmoid(logits)
        output    = dict(y=y, idx=idx, loss=loss, probs=probs)
        if dataloader_idx == 0:
            self.validation_step_outputs.append(output)
        if dataloader_idx == 1:
            self.training_no_augment_step_outputs.append(output)
        return output
               
    def unpack_step_outputs(self, step_outs):
        y     = torch.cat(list(map(lambda x: x['y'], step_outs)), dim=0)
        idx   = torch.cat(list(map(lambda x: x['idx'], step_outs)), dim=0)
        probs = torch.cat(list(map(lambda x: x['probs'], step_outs)), dim=0)
        loss  = torch.stack(list(map(lambda x: x['loss'], step_outs)))
        return y, idx, probs, loss
      
    def on_train_epoch_end(self):
        y, idx, probs, loss = self.unpack_step_outputs(self.training_step_outputs)
        self.training_step_outputs.clear()

        # if self.cfg.nrl.use_ema:
        #     self.ema.update(self.model, probs, self.current_epoch)

        # turn off determinism for metric calculation (not possible otherwise)
        torch.use_deterministic_algorithms(False)
        output = self.train_metrics(probs, y.long())
        torch.use_deterministic_algorithms(True)
        self.log_metrics(loss, output, 'train')
        
        self.probs_train = probs[torch.sort(idx).indices].clone()

        if self.cfg.nrl.co_teaching:

            _, _, probs2, loss2 = self.unpack_step_outputs(self.training_step_outputs2)
            self.training_step_outputs2.clear()

            # turn off determinism for metric calculation (not possible otherwise)
            # torch.use_deterministic_algorithms(False)
            # output2 = self.train_metrics(probs2, y.long())
            # torch.use_deterministic_algorithms(True)
            # self.log_metrics(loss2, output2, 'train_model2')
            
            self.probs_train2 = probs2[torch.sort(idx).indices].clone()

        if self.cfg.nrl.use_ema and self.current_epoch == 0:
            self.probs_train_ema = self.probs_train.clone()

        if self.cfg.tracking.should_track_train_probs:
            self.track_probs(idx, probs, 'train')

        self.update_probabilities()

    def on_validation_epoch_end(self):
        y, idx, probs, loss = self.unpack_step_outputs(self.validation_step_outputs)
        self.validation_step_outputs.clear()

        # turn off determinism for metric calculation (not possible otherwise)
        torch.use_deterministic_algorithms(False)
        output = self.val_metrics(probs, y.long())
        torch.use_deterministic_algorithms(True)
        self.log_metrics(loss, output, 'val')

        if self.cfg.tracking.should_track_val_probs and not self.trainer.sanity_checking:
            self.track_probs(idx, probs, 'val')

        if self.cfg.nrl.use_strong_weak:
            y, idx, probs, loss = self.unpack_step_outputs(self.training_no_augment_step_outputs)
            self.training_no_augment_step_outputs.clear()
            probs[torch.sort(idx).indices]
            self.probs_train = probs

    def NRL_thresh(self, y, probs):
        # Ensure inputs are tensors and moved to the correct device
        y = torch.as_tensor(y, device=self.device, dtype=torch.float32)
        probs = torch.as_tensor(probs, device=self.device, dtype=torch.float32)

        weight = torch.ones_like(y)

        mask_zero = y == 0
        mask_one = y == 1

        # Calculate class sizes and masks for noisy labels
        class_size = y.sum(dim=0) / len(y)
        mask_classes_sub = class_size > self.cfg.nrl.class_min_sub
        mask_classes_add = class_size > self.cfg.nrl.class_min_add

        # Correct noisy zeros (subtractive noise)
        zero_high_prob = (probs > self.cfg.nrl.p1_zero) & mask_zero & mask_classes_sub[None]
        y[zero_high_prob] = 1

        # Dynamic thresholds
        if self.cfg.nrl.use_dynamic_thresholds:
            p2_zero_dynamic = self.cfg.nrl.slope * class_size + self.cfg.nrl.intersect
            zero_mid_prob = (probs > p2_zero_dynamic) & (probs <= self.cfg.nrl.p1_zero) & mask_zero & mask_classes_sub[None]
        else:
            zero_mid_prob = (probs > self.cfg.nrl.p2_zero) & (probs <= self.cfg.nrl.p1_zero) & mask_zero & mask_classes_sub[None]
        weight[zero_mid_prob] = 0

        # Correct noisy ones (additive noise)
        ones_distribution = y.sum(dim=0) / len(y)
        if self.cfg.nrl.p1_one == 'cw':
            thresh_p1 = ones_distribution / 3 - 0.05
        else:
            thresh_p1 = self.cfg.nrl.p1_one
        if self.cfg.nrl.p2_one == 'cw':
            thresh_p2 = ones_distribution - 0.05
        else:
            thresh_p2 = self.cfg.nrl.p2_one

        one_low_prob = (probs < thresh_p1) & mask_one & mask_classes_add[None]
        y[one_low_prob] = 0
        one_mid_prob = (probs < thresh_p2) & (probs >= thresh_p1) & mask_one & mask_classes_add[None]
        weight[one_mid_prob] = 0

        return y.cpu().numpy(), weight.cpu().numpy()

    def NRL_oracle(self, y_gt, yn, probs):
        # accept numpy or tensor, move to the current device
        def to_tensor(el):
            return el if torch.is_tensor(el) else torch.from_numpy(el)

        yn    = to_tensor(yn.copy() if not torch.is_tensor(yn) else yn).to(self.device).float()
        y_gt  = to_tensor(y_gt).to(self.device).float()
        probs = to_tensor(probs).to(self.device).float()

        n_cl = yn.shape[1]
        weight = torch.ones_like(yn, device=self.device)

        mask_noisy_zero = (yn != y_gt) & (yn == 0)
        mask_clean_zero = (yn == y_gt) & (yn == 0)

        mask_noisy_one  = (yn != y_gt) & (yn == 1)
        mask_clean_one  = (yn == y_gt) & (yn == 1)

        # noisy zero correction
        if self.cfg.nrl.oracle_type_noisy_zero == 'ignore':
            weight[mask_noisy_zero] = 0
        elif self.cfg.nrl.oracle_type_noisy_zero == 'flip':
            yn[mask_noisy_zero] = 1

        # noisy one correction
        if self.cfg.nrl.oracle_type_noisy_one == 'ignore':
            weight[mask_noisy_one] = 0
        elif self.cfg.nrl.oracle_type_noisy_one == 'flip':
            yn[mask_noisy_one] = 0

        # clean zero distortion
        if getattr(self.cfg.nrl, 'oracle_type_clean_zero', 'none') != 'none':
            for c in range(n_cl):
                # number of noisy zeros in class c
                k = torch.sum(mask_noisy_zero[:, c]).item()
                if k == 0:
                    continue
                probs_clean_zero = probs[:, c][mask_clean_zero[:, c]]
                # select top k confident zeros to ignore or flip
                _, idc_top_k = torch.sort(probs_clean_zero, descending=True)
                idc_top_k = idc_top_k[:k]
                mask_top_k = torch.zeros(probs_clean_zero.shape[0], device=self.device, dtype=weight.dtype)
                mask_top_k[idc_top_k] = 1.0

                if self.cfg.nrl.oracle_type_clean_zero == 'ignore':
                    weight[mask_clean_zero[:, c], c] = (1.0 - mask_top_k).type_as(weight)
                elif self.cfg.nrl.oracle_type_clean_zero == 'flip':
                    yn[mask_clean_zero[:, c], c] = mask_top_k.type_as(yn)

        # clean one distortion
        if getattr(self.cfg.nrl, 'oracle_type_clean_one', 'none') != 'none':
            for c in range(n_cl):
                # number of noisy ones in class c
                k = torch.sum(mask_noisy_one[:, c]).item()
                if k == 0:
                    continue
                probs_clean_one = probs[:, c][mask_clean_one[:, c]]
                # select bottom k confident ones to ignore or flip
                _, idc_bottom_k = torch.sort(probs_clean_one, descending=False)
                idc_bottom_k = idc_bottom_k[:k]
                mask_bottom_k = torch.zeros(probs_clean_one.shape[0], device=self.device, dtype=weight.dtype)
                mask_bottom_k[idc_bottom_k] = 1.0

                if self.cfg.nrl.oracle_type_clean_one == 'ignore':
                    weight[mask_clean_one[:, c], c] = (1.0 - mask_bottom_k).type_as(weight)
                elif self.cfg.nrl.oracle_type_clean_one == 'flip':
                    yn[mask_clean_one[:, c], c] = (1.0 - mask_bottom_k).type_as(yn)

        return yn.detach().cpu().numpy(), weight.detach().cpu().numpy()

    def get_noise_statistics(self, y, y_gt, y_new, weight):
        
        def to_tensor(el):
            return el if torch.is_tensor(el) else torch.from_numpy(el)
        
        y_new = to_tensor(y_new)
        y = to_tensor(y)
        y_gt = to_tensor(y_gt)
        
        mask_noisy = y != y_gt
        
        added_noise = (y_new != y_gt) * (mask_noisy == 0)       # now noisy         &   previously not noisy
        removed_noise = (y_new == y_gt) * (mask_noisy == 1)     # now not noisy     &   previously noisy
        
        ignored_no_noise = (weight == 0) * (mask_noisy == 0)    # ignore            &   previously not noisy
        ignored_noisy = (weight == 0) * (mask_noisy == 1)       # ignore            &   previously noisy
                    
        noise_statistics = {
            'total': len(y),
            'changed': torch.sum(y_new != y).item(),
            'noise before': torch.sum(mask_noisy).item(),
            'noise now LS-1': torch.sum((y_new != y_gt) * (weight == 1)).item(),
            'noise added': torch.sum(added_noise).item(),
            'noise removed': torch.sum(removed_noise).item(),
            'ignored no noise': torch.sum(ignored_no_noise).item(),
            'ignored noise': torch.sum(ignored_noisy).item()
        }
        return noise_statistics

    def configure_optimizers(self):
        
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.optim.min_lr,
            weight_decay=self.cfg.optim.weight_decay
        )

        max_intervals = int(self.trainer.max_epochs * len(self.datamodule.trainset) / self.cfg.params.batch_size)
        warmup = 10000 if max_intervals > 10000 else 100 if max_intervals > 100 else 0

        lr_scheduler = {'scheduler': LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=warmup,
            max_epochs=max_intervals,
            warmup_start_lr=self.cfg.optim.min_lr / 10,
            eta_min=self.cfg.optim.min_lr / 10
        ), 'name': 'learning_rate', 'interval': "step", 'frequency': 1
        }
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        # if self.cfg.optim.use_ema:
        #     self.ema.update_parameters(self.model)

    def init_metrics(self, num_classes):
        metrics = MetricCollection({
            'APmic'     : AveragePrecision(num_labels=num_classes, task='multilabel', average='micro'),
            'APmac'     : AveragePrecision(num_labels=num_classes, task='multilabel', average='macro'),
            'APclasses' : AveragePrecision(num_labels=num_classes, task='multilabel', average=None),
            'f1mic'     : F1Score(num_labels=num_classes, task='multilabel', average='micro', threshold=0.5),
            'f1mac'     : F1Score(num_labels=num_classes, task='multilabel', average='macro', threshold=0.5),
            'f1classes' : F1Score(num_labels=num_classes, task='multilabel', average=None, threshold=0.5)
        })
        return metrics

    #################
    # LOGGING MODULE
    #################

    def track_probs(self, idx, probs, log_str='train'):
        self.log_inferences(idx, probs, log_str)

    def track_labels(self, labels, col_name='epoch'):
        self.log_inferences(torch.arange(len(labels)), torch.from_numpy(labels), file_ending='labels', col_name=col_name)

    def track_weights(self, weights, col_name='epoch'):
        def to_tensor(el):
            return el if torch.is_tensor(el) else torch.from_numpy(el)
        self.log_inferences(torch.arange(len(weights)), to_tensor(weights), file_ending='weights', col_name=col_name)

    def log_inferences(self, idx, probs, file_ending, col_name='epoch'):
        if col_name == 'epoch':
            col_name = str(self.current_epoch)
        idx = list(idx.cpu().numpy())
        probs = list(map(list, probs.detach().cpu().numpy()))
        df = pd.DataFrame(index=idx, data={col_name: probs})
        path = os.path.join(self.logger.log_dir, 'tracking_{}.parquet'.format(file_ending))
        if os.path.exists(path):
            df_old = pd.read_parquet(path)
            df = pd.concat([df_old, df], axis=1)
        df.to_parquet(path)

    def log_list(self, prefix, metric_list):
        for cl_idx, cl_value in zip(np.arange(self.cfg.dataset.num_classes), metric_list):
            self.log('{}{}'.format(prefix, cl_idx), cl_value, on_epoch=True, on_step=False)

    def log_parquet(self, index, data, name):
        df = pd.DataFrame(index=index, data=data)
        if df.columns.dtype == int:
            df.columns = df.columns.astype(str) 
        path = os.path.join(self.logger.log_dir, name)
        df.to_parquet(path)

    def log_metrics(self, loss, output, log_str='train'):
        ap_classes = output.pop('{}_APclasses'.format(log_str))
        f1_classes = output.pop('{}_f1classes'.format(log_str))

        self.log('{}_loss'.format(log_str), loss.mean(), prog_bar=True)
        self.log_dict(output)
        self.log_list('{}_AP_cl'.format(log_str), ap_classes)
        self.log_list('{}_f1_cl'.format(log_str), f1_classes)

    def log_noisy_labels(self):
        noisy_label = self.datamodule.trainset.labels
        patch_ids = self.datamodule.trainset.patch_names
        idx = np.arange(len(noisy_label))
        data = {'patch_id': patch_ids, 'labels': list(map(list, noisy_label))}
        self.log_parquet(idx, data, 'noisy_labels.parquet')
  
    def log_noise_statistics(self, noise_statistics, file_idx=''):
        df_noise_statistics = pd.DataFrame(noise_statistics, index=pd.Index([self.current_epoch]))
        path = os.path.join(self.logger.log_dir, f'noise_statistics{file_idx}.csv')
        df_noise_statistics.to_csv(path, mode='a', header=not os.path.exists(path))

    def check_running_stats(self):
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d):
                print(f"{name}: track_running_stats={module.track_running_stats}, training={module.training}")

    def disable_track_running_stats(self):
        for module in self.modules():
            if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d):
                module.eval()  # Ensures the layer uses stored stats
                module.track_running_stats = False  # Prevents further updates

    def enable_track_running_stats(self):
        for module in self.modules():
            if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d):
                module.train()
                module.track_running_stats = True

    def get_subtractive_flip_statistics(self, y_start, y_gt, y_new):
        """
        Track 0->1 flips (subtractive-noise corrections) and separate them into:
         - noisy flips: originally incorrect zeros that became 1
         - clean flips: originally correct zeros that became 1 by mistake

        Returns both counts and percentages.
        """

        def to_tensor(el):
            return el if torch.is_tensor(el) else torch.from_numpy(el)

        y_start = to_tensor(y_start).float()
        y_gt    = to_tensor(y_gt).float()
        y_new   = to_tensor(y_new).float()

        # All entries that were originally zero
        start_zero = (y_start == 0)

        # All actual 0->1 flips performed by NRL
        flipped_0_to_1 = start_zero & (y_new == 1)

        # Among originally-zero entries:
        # noisy zero = observed 0 but GT is 1  -> false negative / subtractive noise
        noisy_zero = start_zero & (y_gt == 1)

        # clean zero = observed 0 and GT is 0
        clean_zero = start_zero & (y_gt == 0)

        # Split actual flips into noisy vs clean
        noisy_flips = flipped_0_to_1 & noisy_zero
        clean_flips = flipped_0_to_1 & clean_zero

        n_noisy_flips = noisy_flips.sum().item()
        n_clean_flips = clean_flips.sum().item()
        n_total_flips = n_noisy_flips + n_clean_flips

        n_noisy_zero_total = noisy_zero.sum().item()
        n_clean_zero_total = clean_zero.sum().item()

        # Share among all performed subtractive flips
        pct_noisy_among_flips = 100.0 * n_noisy_flips / n_total_flips if n_total_flips > 0 else 0.0
        pct_clean_among_flips = 100.0 * n_clean_flips / n_total_flips if n_total_flips > 0 else 0.0

        # Optional: recovery/error rates relative to available pools
        pct_noisy_zero_flipped = 100.0 * n_noisy_flips / n_noisy_zero_total if n_noisy_zero_total > 0 else 0.0
        pct_clean_zero_flipped = 100.0 * n_clean_flips / n_clean_zero_total if n_clean_zero_total > 0 else 0.0

        return {
            "subtractive_total_flips": n_total_flips,
            "subtractive_noisy_flips": n_noisy_flips,
            "subtractive_clean_flips": n_clean_flips,

            # these two add to 100%
            "subtractive_pct_noisy_among_flips": pct_noisy_among_flips,
            "subtractive_pct_clean_among_flips": pct_clean_among_flips,

            # optional denominators for interpretation
            "subtractive_total_noisy_zero_pool": n_noisy_zero_total,
            "subtractive_total_clean_zero_pool": n_clean_zero_total,
            "subtractive_pct_noisy_zero_flipped": pct_noisy_zero_flipped,
            "subtractive_pct_clean_zero_flipped": pct_clean_zero_flipped,
        }

    def log_subtractive_flip_statistics(self, stats, file_idx=''):
        df_stats = pd.DataFrame(stats, index=pd.Index([self.current_epoch]))
        path = os.path.join(self.logger.log_dir, f'subtractive_flip_statistics{file_idx}.csv')
        df_stats.to_csv(path, mode='a', header=not os.path.exists(path))

    ####################
    # DATA RELATED HOOKS
    ####################

    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        if self.cfg.nrl.use_strong_weak:
            return [self.datamodule.val_dataloader(), self.datamodule.train_dataloader_no_augment()]
        else:
            return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()

    def train_dataloader_unshuffle(self):
        return self.datamodule.train_dataloader(shuffle=False)

    def train_dataloader_no_augment(self):
        return self.datamodule.train_dataloader_no_augment(shuffle=True)
