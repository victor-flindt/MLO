from data import mnist
import random

import matplotlib.pyplot as plt

train,_ = mnist()


images = train["images"]
labels = train["labels"]
count = len(images)

plt.imshow(images[0],cmap='gray')
plt.title(labels[0])
plt.show()
# fig, axs = plt.subplots(2, 2)
# rand1=random.randint(0, 4999)
# rand2=random.randint(0, 4999)
# rand3=random.randint(0, 4999)
# rand4=random.randint(0, 4999)
# axs[0, 0].imshow(images[rand1],cmap='gray')
# axs[0, 0].set_title(labels[rand1])
# axs[0, 1].imshow(images[rand2],cmap='gray')
# axs[0, 1].set_title(labels[rand2])
# axs[1, 0].imshow(images[rand3],cmap='gray')
# axs[1, 0].set_title(labels[rand3])
# axs[1, 1].imshow(images[rand4],cmap='gray')
# axs[1, 1].set_title(labels[rand4])
#plt.show()
