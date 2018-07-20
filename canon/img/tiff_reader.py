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

    def remove_background(self, cutoff=0.0001):
        self.__reader.fill_black(self.__image)
        self.__image = subtract(self.__image, cutoff)

    def normalize(self, imin=2000, imax=10000):
        img = self.__image
        flatten = np.array(img.reshape(np.prod(img.shape))).clip(0, img.max())
        if np.where(flatten > 0)[0].shape[0] < 100:
            img2 = np.log1p(flatten).reshape(img.shape)
            if img2.max() > 0:
                img2 = 255. * img2 / img2.max()
        else:
            low = np.percentile(flatten[np.where(flatten > 0)], 30)
            high = np.percentile(flatten[np.where(flatten > 0)], 70)
            flatten = flatten.clip(low, flatten.max()) - low
            img2 = flatten.reshape(img.shape)
            img2[np.where(img > high)] = 90. + 10. * (img[np.where(img > high)] - high) / (img.max() - high)
            img2[np.where(img <= high)] = 90. * img2[np.where(img <= high)] / high
            img2 = 2.25 * img2
        self.__image = img2[10:981+10, :]

    def find_peaks(self, npeaks=float('inf')):
        assert npeaks == float('inf') or isinstance(npeaks, int)
        return find_peaks(self.__image, npeaks)
