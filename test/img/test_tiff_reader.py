import unittest
import canon
import numpy as np
from .. import resource


class TiffReaderTestCase(unittest.TestCase):

    @staticmethod
    def ccd_generator():
        for ccd in ('pilatus',):
            yield ccd

    @staticmethod
    def __pilatus_loadPillar5():
        reader = canon.TiffReader('pilatus')
        reader.loadtiff(resource('Pillar5_00001.tif'))
        return reader

    def test_badccd(self):
        self.assertRaises(canon.CanonException, canon.TiffReader, 'bad ccd')

    def test_goodccd(self):
        for ccd in TiffReaderTestCase.ccd_generator():
            canon.TiffReader(ccd)

    def test_load_tif(self):
        reader = TiffReaderTestCase.__pilatus_loadPillar5()
        img = reader.image()
        self.assertTrue(np.array_equal(img.shape, np.array([1043, 981])), "The image size should be 1043 x 981")

    def test_fill_black(self):
        reader = TiffReaderTestCase.__pilatus_loadPillar5()

        img = reader.image()
        original_black_spots = img.shape[0] * img.shape[1] - np.count_nonzero(img)

        reader.fill_black()

        img = reader.image()
        final_black_spots = img.shape[0] * img.shape[1] - np.count_nonzero(img)
        self.assertLess(final_black_spots, original_black_spots, "Number of black spots should be reduced")
        self.assertLess(final_black_spots, 100, "Final number of black spots should be fewer than 100")

if __name__ == '__main__':
    unittest.main()
