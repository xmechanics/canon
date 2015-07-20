import unittest
import numpy as np

import canon
from .. import resource

_TEST_TIFF = 'test00001.tiff'
_TEST_TIFF_2 = 'peak_martensite.tif'
_TEST_DAT = "test.DAT"
_N_SAME_PEAKS = 15


class TiffReaderTestCase(unittest.TestCase):
    _multiprocess_can_split_ = True

    @staticmethod
    def __pilatus_loadPillar5():
        reader = canon.TiffReader('pilatus')
        reader.loadtiff(resource(_TEST_TIFF))
        return reader

    @staticmethod
    def __peaks_in_dat(dat):
        f = open(dat)
        f.readline()
        peaks = []
        for line in f:
            tokens = map(float, line.split())
            peaks.append([1043 - tokens[0], tokens[1]])
        f.close()
        peaks.sort(key=lambda p: p[0])
        peaks = np.array(peaks)
        return peaks

    def setUp(self):
        reader = canon.TiffReader('pilatus')
        reader.loadtiff(resource(_TEST_TIFF))

        self.reader = reader

    def test_badccd(self):
        self.assertRaises(canon.CanonException, canon.TiffReader, 'bad ccd')

    def test_goodccd(self):
        for ccd in ('pilatus',):
            canon.TiffReader(ccd)
            self.assertTrue(True)

    def test_load_tif(self):
        img = self.reader.image()
        self.assertTrue(np.array_equal(img.shape, np.array([1043, 981])), "The image size should be 1043 x 981")

    def test_fill_black(self):
        img = self.reader.image()
        original_black_spots = img.shape[0] * img.shape[1] - np.count_nonzero(img)

        self.reader.fill_black()

        img = self.reader.image()
        final_black_spots = img.shape[0] * img.shape[1] - np.count_nonzero(img)
        self.assertLess(final_black_spots, original_black_spots, "Number of black spots should be reduced")
        self.assertLess(final_black_spots, 100, "Final number of black spots should be fewer than 100")

    def test_find_peaks(self):
        for npeaks in (15, 10, 5):
            yield self.check_peaks, npeaks

    def check_peaks(self, npeaks):
        self.reader.fill_black()
        self.reader.remove_background()
        peaks = self.reader.find_peaks(npeaks)
        self.assertLess(abs(len(peaks) - npeaks), 2, "asked for %d peaks, but found %d" % (npeaks, len(peaks)))

    def test_find_peaks_compare(self):
        self.reader.fill_black()
        self.reader.remove_background()
        peaks = self.reader.find_peaks()

        peaks2 = TiffReaderTestCase.__peaks_in_dat(resource(_TEST_DAT))

        same_peaks = TiffReaderTestCase.__same_peaks(peaks, peaks2)
        n_same_peaks = len(list(same_peaks))

        self.assertGreaterEqual(n_same_peaks, _N_SAME_PEAKS,
                                "should have at least %d peaks close to Nobu's algo." % _N_SAME_PEAKS)

    def test_find_many_peaks(self):
        reader = canon.TiffReader('pilatus')
        reader.loadtiff(resource(_TEST_TIFF_2))
        reader.fill_black()
        reader.remove_background()
        peaks = reader.find_peaks()
        self.assertGreater(len(peaks), 50, "should find more than 50 peaks in martensite pattern")

    @staticmethod
    def __same_peaks(peaks, peaks2):
        def dist(p1, p2):
            return np.linalg.norm(p1 - p2)

        i1 = i2 = 0

        while i1 < len(peaks) and i2 < len(peaks2):
            p1 = peaks[i1]
            p2 = peaks2[i2]

            if dist(p1, p2) < 3:
                i1 += 1
                i2 += 1
                yield p1, p2
            else:
                if p2[0] < p1[0] - 1:
                    i2 += 1
                else:
                    i1 += 1

if __name__ == '__main__':
    unittest.main()
