# Import the `transform` module from `skimage`
from skimage import transform
from skimage import color
import numpy as np


def resize(images, w=28, h=28):
    # Rescale the images in the `images` array
    return [transform.resize(image, (w, h)) for image in images]

def rgb2gray(images):
    # Convert `images` to grayscale
    return color.rgb2gray(np.array(images))