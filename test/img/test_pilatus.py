import pytest
import canon
import numpy as np
from .. import resource
from canon.img.pilatus import Pilatus


def test_fill_black(pilatus):
    assert isinstance(pilatus, canon.TiffReader)

    pilatus.loadtiff(resource("test00001.tiff"))

    img = pilatus.image()
    original_black_spots = img.shape[0] * img.shape[1] - np.count_nonzero(img)

    Pilatus.fill_black(img)
    final_black_spots = img.shape[0] * img.shape[1] - np.count_nonzero(img)

    # Number of black spots should be reduced
    assert final_black_spots < original_black_spots

    # Final number of black spots should be fewer than 100
    assert final_black_spots < 100

if __name__ == '__main__':
    pytest.main()
