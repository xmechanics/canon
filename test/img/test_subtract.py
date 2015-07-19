import unittest
import numpy as np
from PIL import Image
from canon.img.subtract_bg import subtract
from .. import resource


class SubtractTestCase(unittest.TestCase):

    def test_subtract(self):
        image = np.array(Image.open(resource('test00001.tiff')))
        img_sub = subtract(image, 0.001)
        self.assertLessEqual(np.max(img_sub), np.max(image), "subtracting background should reduce max intensity")
        self.assertGreaterEqual(np.min(img_sub), np.min(image), "subtracting background should raise min intensity")

if __name__ == '__main__':
    unittest.main()
