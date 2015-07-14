from ..__imports import *
from skimage import exposure


def subtract(image, cutoff):
    assert isinstance(image, np.ndarray)
    assert cutoff < 1

    threshold = __bg_threshold(image, cutoff)

    return np.where(image < threshold, 0, image)


def extract_bg(img, cutoff):
    threshold = __bg_threshold(img, cutoff)
    return np.where(img < threshold, img, 0)


def __bg_threshold(imgdata, cutoff):
    cdf, bins = exposure.cumulative_distribution(imgdata, nbins=10000)

    for c, b in zip(cdf, bins):
        if c > 1 - cutoff:
            return b

