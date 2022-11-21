import os
import imageio.v3 as iio
from pathlib import Path
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image as im

imagesUni = list()
imagesStruc = list()
imagesGt = list()

mainPath = "/Users/johanna/Desktop/Oberseminar/synthetic_data"
uniPath = mainPath + '/uniform'
strucPath = mainPath + '/structured'
gtPath = mainPath + '/ground_truth'

nrAugmentations = 2


for file in Path(uniPath).iterdir():
    if not file.is_file():
        continue

    imagesUni.append(iio.imread(file))

for file in Path(strucPath).iterdir():
    if not file.is_file():
        continue

    imagesStruc.append(iio.imread(file))

for file in Path(gtPath).iterdir():
    if not file.is_file():
        continue

    imagesGt.append(iio.imread(file))

#pipeline
ia.seed(1)

# Example batch of images.
# The array has shape (32, 64, 64, 3) and dtype uint8.
#images = np.array(
 #   [ia.quokka(size=(64, 64)) for _ in range(32)],
  #  dtype=np.uint8
#)

def augParameters():
    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Crop(percent=(0, 0.1)), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
        ], random_order=True) # apply augmenters in random order
    return seq

uniAug = []
strucAug = []
gtAug = []

for l in range(0, nrAugmentations):
    newSeed = augParameters()
    for i in range(0,len(imagesUni)):
        augseq_det = newSeed.to_deterministic()
        imageArray = [imagesUni[i],imagesStruc[i],imagesGt[i]]
        aframes_aug = [augseq_det.augment_image(frame) for frame in imageArray]
        #augArray = newSeed(images=imageArray)
        uniAug.append(aframes_aug[0])
        strucAug.append(aframes_aug[1])
        gtAug.append(aframes_aug[2])


im.fromarray(uniAug[0]).show()

safePathUni = '/Users/johanna/Desktop/Oberseminar/augmented_data/uni_aug/'
safePathStruc = '/Users/johanna/Desktop/Oberseminar/augmented_data/struc_aug/'
safePathGt = '/Users/johanna/Desktop/Oberseminar/augmented_data/gt_aug/'

#safe augmented images in directory with matching names
for k in range(0,len(uniAug)):
    uniName = "uni_aug_" + str(k) + '.png'
    strucName = "struc_aug_" + str(k) + '.png'
    gtName = "gt_aug_" + str(k) + '.png'

    im.fromarray(uniAug[k]).save(safePathUni + uniName, 'PNG')
    im.fromarray(strucAug[k]).save(safePathStruc + strucName, 'PNG')
    im.fromarray(gtAug[k]).save(safePathGt + gtName, 'PNG')
