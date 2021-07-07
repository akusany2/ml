import matplotlib.pyplot as plt
import numpy as np


PATH = "./model_net.pth"

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images


# dataiter = iter(load_data.trainloader)
# images, labels = dataiter.next()

# show images
# imshow(torchvision.utils.make_grid(images))
# print labels
# print(
#     " ".join("%5s" % load_data.classes[labels[j]] for j in range(load_data.batch_size))
# )
