import unittest
import pytest
import numpy as np

import canon
from .. import resource

_N_SAME_PEAKS = 15


class TiffReaderTestCase(unittest.TestCase):
    _multiprocess_can_split_ = True

    @staticmethod
    def __pilatus_load_test1():
        reader = canon.TiffReader('pilatus')
        reader.loadtiff(resource('test00001.tiff'))
        return reader

    @staticmethod
    def __peaks_in_dat(dat):
        f = open(dat)
        f.readline()
        peaks = []
        for line in f:
            tokens = list(map(float, line.split()))
            peaks.append([1043 - tokens[0], tokens[1]])
        f.close()
        peaks.sort(key=lambda p: p[0])
        peaks = np.array(peaks)
        return peaks

    def test_badccd(self):
        self.assertRaises(canon.CanonException, canon.TiffReader, 'bad ccd')

    def test_goodccd(self):
        for ccd in ('pilatus',):
            canon.TiffReader(ccd)
            self.assertTrue(True)

    def test_load_tif(self):
        reader = TiffReaderTestCase.__pilatus_load_test1()
        img = reader.image()
        self.assertTrue(np.array_equal(img.shape, np.array([1043, 981])), "The image size should be 1043 x 981")

    def test_find_peaks(self):
        for npeaks in (15, 10, 5):
            yield self.check_peaks, npeaks

    def check_peaks(self, npeaks):
        reader = TiffReaderTestCase.__pilatus_load_test1()
        reader.remove_background()
        peaks = reader.find_peaks(npeaks)
        self.assertLess(abs(len(peaks) - npeaks), 2, "asked for %d peaks, but found %d" % (npeaks, len(peaks)))

    def test_find_peaks_compare(self):
        reader = TiffReaderTestCase.__pilatus_load_test1()
        reader.remove_background()
        peaks = reader.find_peaks()

        peaks2 = TiffReaderTestCase.__peaks_in_dat(resource("test.DAT"))

        same_peaks = TiffReaderTestCase.__same_peaks(peaks, peaks2)
        n_same_peaks = len(list(same_peaks))

        # self.assertGreaterEqual(n_same_peaks, _N_SAME_PEAKS,
        #                         "should have at least %d peaks close to Nobu's algo." % _N_SAME_PEAKS)

    def test_find_many_peaks(self):
        reader = canon.TiffReader('pilatus')
        reader.loadtiff(resource('peak_martensite.tif'))
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
    pytest.main()
