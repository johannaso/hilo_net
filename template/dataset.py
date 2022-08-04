import os
import glob

from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, struct_dir, homogen_dir, gt_dir, scale=1):
        self.struct_dir = struct_dir # hier liegen die strukturierten Eingangsbilder
        self.homogen_dir = homogen_dir # hier liegen die homogen-beleuchteten Eingangsbilder
        self.gt_dir = gt_dir # hier liegen die Ausgangsbilder
        self.scale = scale

        try:
            self.ids = [s.split('.')[0] for s in os.listdir(self.struct_dir)] # Liste erzeugen mit allen Datenelementen
        except FileNotFoundError:
            self.ids = []

    def __len__(self):
        return len(self.ids) #Gesamtanzahl an Datenelementen

    def preprocess(self, img): #To Do: Funktion anpassen, um Eingangsbilder ggf. anzupassen (img --> Eingangsbild)
        w, h = img.size
        _h = int(h * self.scale)
        _w = int(w * self.scale)
        assert _w > 0
        assert _h > 0

        _img = img.resize((_w, _h))
        _img = np.array(_img)
        if len(_img.shape) == 2:  ## gray/mask images
            _img = np.expand_dims(_img, axis=-1)

        # hwc to chw
        _img = _img.transpose((2, 0, 1))
        if _img.max() > 1:
            _img = _img / 255.
        return _img

    def __getitem__(self, i): #To Do: Funktion anpassen, um ein komplettes Datenelement zu laden (img --> Eingang, mask --> Output)
        idx = self.ids[i]
        img_files = glob.glob(os.path.join(self.img_dir, idx+'.*'))
        mask_files = glob.glob(os.path.join(self.mask_dir, idx+'_mask.*'))

        assert len(img_files) == 1, f'{idx}: {img_files}'
        assert len(mask_files) == 1, f'{idx}: {mask_files}'

        # use Pillow's Image to read .gif mask
        # https://answers.opencv.org/question/185929/how-to-read-gif-in-python/
        img = Image.open(img_files[0])
        mask = Image.open(mask_files[0])
        assert img.size == mask.size, f'{img.shape} # {mask.shape}'

        img = self.preprocess(img)
        mask = self.preprocess(mask)

        return torch.from_numpy(img).float(), \
            torch.from_numpy(mask).float()