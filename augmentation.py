import random
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import imageio
import albumentations as A
from PIL import Image

def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)

picture_path_uniform = 'img/hilo_images/uniform'
picture_path_speckle = 'img/hilo_images/structured'
picture_path_hilo = 'img/hilo_images/ground_truth'
image_list_uniform = os.listdir(picture_path_uniform)
image_list_speckle = os.listdir(picture_path_speckle)
image_list_hilo = os.listdir(picture_path_hilo)

print(image_list_hilo)
print(image_list_uniform)
print(image_list_speckle)

#remove '.DS_Store'
if '.DS_Store' in image_list_hilo:
     image_list_hilo.remove('.DS_Store')
if '.DS_Store' in image_list_speckle:
    image_list_speckle.remove('.DS_Store')
if '.DS_Store' in image_list_uniform:
   image_list_uniform.remove('.DS_Store')
print(image_list_hilo)
print(image_list_uniform)
print(image_list_speckle)

transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RGBShift(p=0.2),
    ],
    additional_targets={'image0': 'image', 'image1': 'image'}
)

save_prefix = 'aug'
save_format = 'jpg'

save_uniform_path = 'img/training/img/img_uniform/augmented'
save_speckle_path = 'img/training/img/img_speckle/augmented'
save_hilo_path = 'img/training/img/img_groundt/augmented'

batch_size = 5


for i in range(395, 564):
    #load one image from each array
    print(i)
    image = cv2.imread(os.path.join(picture_path_uniform, image_list_uniform[i]))
    #image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    visualize(image)
    image0 = cv2.imread(os.path.join(picture_path_speckle, image_list_speckle[i]))
    #image0 = np.array(image0)
    image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
    visualize(image0)
    image1 = cv2.imread(os.path.join(picture_path_hilo, image_list_hilo[i]))
    #image1 = np.array(image1)
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    visualize(image1)

    # make name
    image_uniform_name = image_list_uniform[i].split('.')[0]
    image0_number = image_uniform_name.split('_')[1]
    print(image0_number)
    image_speckle_name = image_list_speckle[i].split('.')[0]
    image1_number = image_speckle_name.split('_')[1]
    image_hilo_name = image_list_hilo[i].split('.')[0]
    image2_number = image_hilo_name.split('_')[1]

    for n in range(batch_size):

        #transform the images at the same time with same parameters
        random.seed(42)
        transformed = transform(image=image, image0=image0, image1=image1)

        #visualize(transformed[image])
        #visualize(transformed[image0])

        #safe them into their directory

        uni_image = save_prefix + '_' + image0_number + '_' + str(n)
        speckle_image = save_prefix + '_' + image1_number + '_' + str(n)
        hilo_image = save_prefix + '_' + image2_number + '_' + str(n)

        uni_aug = transformed['image']
        speckle_aug = transformed['image0']
        hilo_aug = transformed['image1']

        uni_aug = Image.fromarray(uni_aug)
        speckle_aug = Image.fromarray(speckle_aug)
        hilo_aug = Image.fromarray(hilo_aug)

        #uni_aug.save(save_uniform_path + uni_image + '.jpeg')


        uni_aug.save(save_uniform_path + '/' + uni_image + '.jpg', 'JPEG')

        speckle_aug.save(save_speckle_path + '/' + speckle_image + '.jpg', 'JPEG')

        hilo_aug.save(save_hilo_path + '/' + hilo_image + '.jpg', 'JPEG')

