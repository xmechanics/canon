import logging
import numpy as np
from skimage.transform import resize

from canon.pattern.feature_extractor import FeaturesExtractor

_logger = logging.getLogger(__name__)


class RescaleExtractor(FeaturesExtractor):

    def __init__(self, shape):
        FeaturesExtractor.__init__(self)
        self.__shape = shape

    def features(self, img_data, skip_normalize=True):
        if not np.all(np.equal(img_data.shape[1:], self.__shape)):
            img_data = [resize(img, self.__shape, mode='reflect').flatten().astype('float32') for img in img_data]
        return np.array(img_data).astype('float32')
