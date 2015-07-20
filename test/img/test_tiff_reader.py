import unittest
import numpy as np

import canon
from .. import resource


class TiffReaderTestCase(unittest.TestCase):
    _multiprocess_can_split_ = True     # each test has its own fixture

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
        for ccd in ('pilatus',):
            canon.TiffReader(ccd)
            self.assertTrue(True)

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

    def test_find_peaks(self):
        for npeaks in (15, 10, 5):
            self.check_peaks(npeaks)

    def check_peaks(self, npeaks):
        reader = TiffReaderTestCase.__pilatus_loadPillar5()
        reader.fill_black()
        reader.remove_background()
        peaks = reader.find_peaks(npeaks)
        self.assertLess(abs(len(peaks) - npeaks), 2, "asked for %d peaks, but found %d" % (npeaks, len(peaks)))

if __name__ == '__main__':
    unittest.main()
