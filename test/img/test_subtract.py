import unittest
import numpy as np
from PIL import Image
from canon.img.subtract_bg import subtract, extract_bg
from .. import resource


class HistoStateTestCase(unittest.TestCase):

    def test_subtract(self):
        image = np.array(Image.open(resource('Pillar5_00001.tif')))
        img_sub = subtract(image, 0.001)
        self.assertEqual(np.max(img_sub), np.max(image), "subtracting background does not affect max intensity")
        self.assertGreater(np.min(img_sub), np.min(image), "subtracting background should raise min intensity")

    def test_extract(self):
        image = np.array(Image.open(resource('Pillar5_00001.tif')))
        img_sub = extract_bg(image, 0.001)
        self.assertLess(np.max(img_sub), np.max(image), "background should have lower max intensity")
        self.assertEqual(np.min(img_sub), np.min(image), "background should have the same min intensity")

if __name__ == '__main__':
    unittest.main()
