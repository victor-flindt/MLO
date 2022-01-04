
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

## loads the testing data
## sorts images and labels in 2 variables.
test = np.load("corruptmnist/test.npz")
test_images = test["images"]
test_labels = test["labels"]

## 
for index,label in enumerate(test_labels):
    if label == 1:
        im = Image.fromarray(test_images[index])
        plt.imsave(f'test/1/test{index}.jpg', im, cmap='Greys')
    elif label == 2:
        im = Image.fromarray(test_images[index])
        plt.imsave(f'test/2/test{index}.jpg', im, cmap='Greys')
    elif label == 3:
        im = Image.fromarray(test_images[index])
        plt.imsave(f'test/3/test{index}.jpg', im, cmap='Greys')
    elif label == 4:
        im = Image.fromarray(test_images[index])
        plt.imsave(f'test/4/test{index}.jpg', im, cmap='Greys')
    elif label == 5:
        im = Image.fromarray(test_images[index])
        plt.imsave(f'test/5/test{index}.jpg', im, cmap='Greys')
    elif label == 6:
        im = Image.fromarray(test_images[index])
        plt.imsave(f'test/6/test{index}.jpg', im, cmap='Greys')
    elif label == 7:
        im = Image.fromarray(test_images[index])
        plt.imsave(f'test/7/test{index}.jpg', im, cmap='Greys')
    elif label == 8:
        im = Image.fromarray(test_images[index])
        plt.imsave(f'test/8/test{index}.jpg', im, cmap='Greys')
    elif label == 9:
        im = Image.fromarray(test_images[index])
        plt.imsave(f'test/9/test{index}.jpg', im, cmap='Greys')

