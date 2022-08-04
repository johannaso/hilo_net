import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from model import UNet
from dataset import CustomDataset

## Training ##
train_loader = DataLoader(Custom_Dataset(train_data_dir)) #lade das Trainings-Datenset
trainer = pl.trainer(max_epochs = 5) #Wieviele Epochen soll trainiert werden?
model = UNet() #erstelle das Model

trainer.fit(model, train_dataloader=train_loader) #hier passiert das eigentliche Training

## Testing ##
test_loader = DataLoader(Custom_Dataset(test_data_dir)) #lade das Test-Datenset
trainer.test(model, data_loader = test_loader) # Teste das trainierte Modell