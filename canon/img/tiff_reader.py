from skimage.external.tifffile import imread

from canon.__imports import *
from canon.img.subtract_bg import subtract
from canon.img.peaks import find_peaks
from canon.img.pilatus import Pilatus


class TiffReader:
    """
    This is an interface
    """

    PILATUS = "pilatus"

    CCD_READERS = {
        PILATUS: Pilatus
    }

    @classmethod
    def __provide_reader(cls, ccd):
        assert isinstance(ccd, str)

        try:
            ccd = ccd.lower()
            return TiffReader.CCD_READERS[ccd]
        except KeyError as e:
            raise CanonException("Unrecognizable CCD %s" % ccd, e)

    def __init__(self, ccd):
        self.__reader = TiffReader.__provide_reader(ccd)
        self.__image = None

    def loadtiff(self, filename):
        img = imread(filename)
        self.__image = img

    def image(self):
        return self.__image.copy()

    def remove_background(self, cutoff=0.001):
        self.__reader.fill_black(self.__image)
        self.__image = subtract(self.__image, cutoff)

    def normalize(self, cutoff=80):
        img = self.__image
        flat = img.reshape(img.shape[0] * img.shape[1])
        high = np.percentile(flat[np.where(flat > 0)], cutoff)
        img2 = np.zeros(img.shape)
        img2[np.where(img > high)] = 90. + 10. * (img[np.where(img > high)] - high) / (img.max() - high)
        img2[np.where(img <= high)] = 90. * img[np.where(img <= high)] / high
        self.__image = 2.55 * img2

    def find_peaks(self, npeaks=float('inf')):
        assert npeaks == float('inf') or isinstance(npeaks, int)
        return find_peaks(self.__image, npeaks)
