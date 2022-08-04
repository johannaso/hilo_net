from argparse import ArgumentParser

import numpy
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchgeometry as tgm
import image_similarity_measures
import torchvision
from image_similarity_measures.quality_metrics import rmse, psnr, issm
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

import wandb
from dataset import CustomDataset

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 2


class Unet(pl.LightningModule):
    def __init__(self, hparams):
        super(Unet, self).__init__()
        self.save_hyperparameters(hparams)
        self.n_channels = hparams.n_channels
        self.bilinear = True

        #log hyperparameters
        self.save_hyperparameters()

        #compute accuracy
        self.train_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        # self.train_acc = pl.metrics.Accuracy()
        # self.valid_acc = pl.metrics.Accuracy()
        # self.test_acc = pl.metrics.Accuracy()

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def down(in_channels, out_channels):
            return nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_channels, out_channels)
            )

        class up(nn.Module):
            def __init__(self, in_channels, out_channels, bilinear=True):
                super().__init__()

                if bilinear:
                    self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                else:
                    self.up = nn.ConvTranpose2d(in_channels // 2, in_channels // 2,
                                                kernel_size=2, stride=2)

                self.conv = double_conv(in_channels, out_channels)

            def forward(self, x1, x2):
                x1 = self.up(x1)
                diffY = x2.size()[2] - x1.size()[2]
                diffX = x2.size()[3] - x1.size()[3]

                x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])
                x = torch.cat([x2, x1], dim=1)
                return self.conv(x)

        self.inc = double_conv(self.n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        # self.out = up(128*128, 3, 1)
        self.out = nn.Conv2d(64, 3, kernel_size=(3, 3), padding="same")# TODO: Output Layer anpassen! --> Keine Klassen
        # self.out = up(128, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        preds = self(x)
        target = self(y)


        #calculate accuracy
        # accuracy = torchmetrics.Accuracy()
        im1 = numpy.array(torchvision.transforms.ToPILImage()(numpy.squeeze(target)))
        im2 = numpy.array(torchvision.transforms.ToPILImage()(numpy.squeeze(preds)))
        similarity = issm(im1,im2)
        # self.train_acc = torch.tensor(similarity)
        # loss = F.mse_loss(preds,target)
        ssim = tgm.losses.SSIM(5, reduction="sum")
        loss = ssim(preds,target)
        self.log("train/loss", loss)
        # self.train_acc(x,y)
        self.log('train/acc', similarity, on_epoch=True)
        tensorboard_logs = {'train_loss': loss}
        wandb.log({"prediction/train": wandb.Image(preds), "groundtrouth/train": wandb.Image(target)})

        return {'loss': loss, 'log': tensorboard_logs}


    def test_step(self, batch, batch_idx):
        X_batch, Y_batch = batch



        # self.log("test/acc_epoch", self.test_acc, on_step=False, on_epoch=True)
        preds = self(X_batch)
        target = self(Y_batch)
        im1 = torchvision.transforms.ToPILImage()(numpy.squeeze(target))
        im2 = torchvision.transforms.ToPILImage()(numpy.squeeze(preds))
        y_hat = self.forward(X_batch)
        ssim = tgm.losses.SSIM(5, reduction="sum")
        loss_val = ssim(preds, target)
        self.log("test/loss_epoch", loss_val, on_step=False, on_epoch=True)
        wandb.log({"train/loss": loss_val})

        # calculate accuracy
        accuracy = torchmetrics.Accuracy()
        similarity = issm(numpy.asarray(im1),numpy.asarray(im2))
        self.log('test/acc', similarity, on_epoch=True)
        wandb.log({"prediction": wandb.Image(preds), "groundtrouth": wandb.Image(target)})
        return {"loss": loss_val}

    def validation_step(self, batch, batch_nb):
        if batch_nb == 0:
            n = 20
            x, y = batch
            images = [img for img in x[:n]]
            captions = [f'Ground Truth: {y_i} - Prediction: {y_pred}'
                        for y_i, y_pred in zip(y[:n], outputs[:n])]
            WandbLogger.wandb_logger.log_image(
                key='sample_images',
                images=images,
                caption=captions)

        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y) if self.n_classes > 1 else \
            F.binary_cross_entropy_with_logits(y_hat, y) ## To Do: Loss Funktion austauschen
        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean() ## To Do: Loss Funktion austauschen
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=0.1, weight_decay=1e-8)



    def __dataloader(self): #ToDo: Dataloader auf Dataset anpassen
        dataset = self.hparams.dataset
        dataset = CustomDataset(f'./dataset/{dataset}/train', f'./dataset/{dataset}/train_masks')
        n_val = int(len(dataset) * 0.1)
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])
        train_loader = DataLoader(train_ds, batch_size=1, pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1, pin_memory=True, shuffle=False)

        return {
            'train': train_loader,
            'val': val_loader,
        }

    #@pl.data_loader
    #def train_dataloader(self):
     #   return self.__dataloader()['train']

    #@pl.data_loader
    #def val_dataloader(self):
     #   return self.__dataloader()['val']

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument('--n_channels', type=int, default=3)
        return parser
