import torch
from torchvision import transforms, datasets
from PIL import Image, ImageFilter
import numpy as np


def get_data():
    train_set = datasets.MNIST('../dataset', train=True, download=True,
                               transform=transforms.Compose([transforms.ToTensor()]))

    test_set = datasets.MNIST('../dataset', train=False, download=True,
                              transform=transforms.Compose([transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=True)

    return train_loader, test_loader


def prepare_image(path: str):
    """
    Converting image to MNIST dataset format
    """

    im = Image.open(path).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    new_image = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        new_image.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        new_image.paste(img, (wleft, 4))  # paste resized image on white canvas

    pixels = list(new_image.getdata())  # get pixel values
    pixels_normalized = [(255 - x) * 1.0 / 255.0 for x in pixels]

    # Need adequate shape
    adequate_shape = np.reshape(pixels_normalized, (1, 28, 28))
    output = torch.FloatTensor(adequate_shape).unsqueeze(0)
    return output
