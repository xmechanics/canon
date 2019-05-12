import pytest
import numpy as np
from skimage.external.tifffile import imread
from canon.img.subtract_bg import subtract
from .. import resource


def test_subtract():
    image = imread(resource('test00001.tiff'))
    img_sub = subtract(image, 0.001)

    # subtracting background should reduce max intensity
    assert np.max(img_sub) <= np.max(image)
    # subtracting background should raise min intensity
    assert np.min(img_sub) >= np.min(image)

if __name__ == '__main__':
    pytest.main()
