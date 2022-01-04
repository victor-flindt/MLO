
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

## loads the training data
## sorts images and labels in 2 variables.
train = np.load("corruptmnist/train_0.npz")
train_images = train["images"]
train_labels = train["labels"]

## 
for index,label in enumerate(train_labels):
    if label == 1:
        im = Image.fromarray(train_images[index])
        plt.imsave(f'train/1/train{index}.jpg', im, cmap='Greys')
    elif label == 2:
        im = Image.fromarray(train_images[index])
        plt.imsave(f'train/2/train{index}.jpg', im, cmap='Greys')
    elif label == 3:
        im = Image.fromarray(train_images[index])
        plt.imsave(f'train/3/train{index}.jpg', im, cmap='Greys')
    elif label == 4:
        im = Image.fromarray(train_images[index])
        plt.imsave(f'train/4/train{index}.jpg', im, cmap='Greys')
    elif label == 5:
        im = Image.fromarray(train_images[index])
        plt.imsave(f'train/5/train{index}.jpg', im, cmap='Greys')
    elif label == 6:
        im = Image.fromarray(train_images[index])
        plt.imsave(f'train/6/train{index}.jpg', im, cmap='Greys')
    elif label == 7:
        im = Image.fromarray(train_images[index])
        plt.imsave(f'train/7/train{index}.jpg', im, cmap='Greys')
    elif label == 8:
        im = Image.fromarray(train_images[index])
        plt.imsave(f'train/8/train{index}.jpg', im, cmap='Greys')
    elif label == 9:
        im = Image.fromarray(train_images[index])
        plt.imsave(f'train/9/train{index}.jpg', im, cmap='Greys')

