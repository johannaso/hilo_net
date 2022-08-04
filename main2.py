from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader


import dataset
import wandb
from model import Unet

#weights and biases
wandb.login(key="fc48556b8dcb07c14d707ae0dddca9dc992f66eb")
wandb.init(project="HiLo", entity="sophie_jo")
wandb_logger = WandbLogger()




TRAIN_PATH = 'img/training'
TEST_PATH = 'img/test'

uniform_path = TRAIN_PATH + '/img/img_uniform/augmented'
speckle_path = TRAIN_PATH + '/img/img_speckle/augmented'
hilo_path = TRAIN_PATH + '/img/img_groundt/augmented'

uniform_test = TEST_PATH + '/img_uniform'
speckle_test = TEST_PATH + '/img_speckle'
hilo_test = TEST_PATH + '/img_groundt'

IMG_CHANNELS = 3



def main(hparams):

    ## Training ##
    train_loader = DataLoader(
        dataset.CustomDataset(speckle_path, uniform_path, hilo_path))  # lade das Trainings-Datenset
    samples = next(iter(train_loader))
    trainer = pl.Trainer(limit_train_batches=50, max_epochs=5, log_every_n_steps=10, logger=wandb_logger)
    # model = UNet() #erstelle das Model
    model = Unet(hparams)
    wandb_logger.watch(model, log="all")
    trainer.fit(model, train_loader)  # hier passiert das eigentliche Training

    ## Testing ##
    test_loader = DataLoader(dataset.CustomDataset(speckle_test,uniform_test,hilo_test)) #lade das Test-Datenset
    trainer.test(model = model, dataloaders=test_loader) # Teste das trainierte Modell
    # trainer.test(model)
    wandb.finish()

if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--dataset', required=True)
    parent_parser.add_argument('--log_dir', default='lightning_logs')

    parser = Unet.add_model_specific_args(parent_parser)
    hparams = parser.parse_args()

    main(hparams)
