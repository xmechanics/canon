from skimage.external.tifffile import imread

from ..__imports import *
from subtract_bg import subtract
from peaks import find_peaks
from pilatus import Pilatus


class TiffReader:
    """
    This is an interface
    """

    CCD_READERS = {
        'pilatus': Pilatus
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
        img = imread(filename).astype('float')
        self.__image = img

    def image(self):
        return self.__image.copy()

    def remove_background(self, cutoff=0.001):
        self.__reader.fill_black(self.__image)
        self.__image = subtract(self.__image, cutoff)

    def find_peaks(self, npeaks=float('inf')):
        assert npeaks == float('inf') or isinstance(npeaks, int)

        return find_peaks(self.__image, npeaks)
