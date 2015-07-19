import unittest
import canon

from test import resource
from canon.img.peaks import find_peaks


class PeaksTestCase(unittest.TestCase):
    img = None

    @classmethod
    def setUpClass(cls):
        reader = canon.TiffReader('pilatus')
        reader.loadtiff(resource('Pillar5_00001.tif'))
        reader.fill_black()
        reader.remove_background()
        PeaksTestCase.img = reader.image()

    def test_all_peaks(self):
        peaks = find_peaks(PeaksTestCase.img)
        self.assertGreater(len(peaks), 15)

    def test_peaks(self):
        for npeaks in (15, 10, 5):
            self.check_peaks(npeaks)

    def check_peaks(self, npeaks):
        peaks = find_peaks(PeaksTestCase.img, npeaks)
        self.assertLess(abs(len(peaks) - npeaks), 2, "asked for %d peaks, but found %d" % (npeaks, len(peaks)))


if __name__ == '__main__':
    unittest.main()
