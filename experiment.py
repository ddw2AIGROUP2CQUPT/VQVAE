import os
import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

        
    def on_validation_end(self) -> None:
        self.sample_images()
        
    def sample_images(self):
        # Get sample reconstruction image            
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

#         test_input, test_label = batch
        recons = self.model.generate(test_input, labels = test_label)
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir , 
                                       "Reconstructions", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels = test_label)
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir , 
                                           "Samples",      
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass
    # def configure_optimizers(self):
    #     optims = []
    #     scheds = []

    #     # 基础优化器
    #     optimizer = optim.Adam(
    #         self.model.parameters(),
    #         lr=self.params['LR'],
    #         weight_decay=self.params['weight_decay']
    #     )
    #     optims.append(optimizer)


    #     # 只使用 CosineAnnealingLR，无 warmup
    #     try:
    #         total_epochs = 100

    #         cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #             optimizer,
    #             T_max=total_epochs,
    #             eta_min=1e-5
    #         )

    #         scheds.append({
    #             "scheduler": cosine_scheduler,
    #             "interval": "epoch",
    #             "frequency": 1,
    #             "monitor": "loss",
    #             "name": "lr",
    #             "strict": True,
    #             "optimizer": optimizer  # 显式添加 optimizer 以避免 Lightning 报错
    #         })

    #         return optims, scheds
    #     except:
    #         return optims

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        
        except:
            return optims
        
# def configure_optimizers(self):

#         optims = []
#         scheds = []

#         optimizer = optim.Adam(self.model.parameters(),
#                                lr=self.params['LR'],
#                                weight_decay=self.params['weight_decay'])
#         optims.append(optimizer)
     
#             # 获取warmup参数（默认5个epoch）
#         warmup_epochs = 5
    
#         # 线性warmup阶段
#         warmup_scheduler = optim.lr_scheduler.LambdaLR(
#             optims[0],
#             lr_lambda=lambda epoch: min(
#                 (epoch + 1) / warmup_epochs,  # 线性增长
#                 1.0  # 上限
#             )
#         )
        
#         # 余弦退火阶段参数（总epoch 100，warmup 5）
#         cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
#             optims[0],
#             T_max=100 - warmup_epochs,  # 95
#             eta_min=self.params['scheduler_eta_min']  # 1e-5
#         )
        
#         # 组合两个调度器
#         scheduler = optim.lr_scheduler.SequentialLR(
#             optims[0],
#             schedulers=[warmup_scheduler, cosine_scheduler],
#             milestones=[warmup_epochs]
#         )
#         scheduler.optimizer = optims[0]
#         scheds.append({
#             'scheduler': scheduler,
#             'interval': 'epoch',
#             'frequency': 1,
#             'optimizer': optims[0]
#         })

#         return optims, scheds