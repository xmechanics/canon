from canon.__imports import *
from skimage import exposure


def subtract(image, cutoff):
    assert isinstance(image, np.ndarray)
    assert cutoff < 1

    bg = __extract_bg(image, cutoff)
    img2 = image - bg
    ratio = np.where(img2 > 0)[0].sum() / np.where(image > 0)[0].sum()
    while ratio < 1e-3:
        cutoff = cutoff * 2
        bg = __extract_bg(image, cutoff)
        img2 = image - bg
        ratio = np.where(img2 > 0)[0].sum() / np.where(image > 0)[0].sum()

    #print(bg.max(), image.max(), img2.mean(), image.mean(), np.where(img2 > 0)[0].sum(), np.where(image > 0)[0].sum())
    # print(img2.mean() / image.mean(), np.where(img2 > 0)[0].sum() / np.where(image > 0)[0].sum())
    # while bg.max() >= 0.5*image.max():
    #     cutoff = cutoff * 2
    #     bg = __extract_bg(image, cutoff)

    return img2


def __extract_bg(img, cutoff):
    hist, bins = exposure.histogram(img, nbins=100000)
    histcuftoff = np.max(hist) * cutoff
    bg_min, bg_max = np.inf, 0

    for h, b in zip(hist, bins):
        if h >= histcuftoff:
            bg_min = min(bg_min, b)
            bg_max = max(bg_max, b)

    if bg_max == 0:
        bg_max = 100. * histcuftoff

    if bg_min > bg_max:
        bg_min, bg_max = bg_max, bg_min

    bg = np.where(img < bg_min, bg_min, img)
    bg = np.where(bg > bg_max, bg_max, bg)

    return bg

