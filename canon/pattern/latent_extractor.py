import logging
import os
import numpy as np
from skimage.transform import resize

from canon.pattern.feature_extractor import FeaturesExtractor
from canon.autoencode.models import load_encoder

_logger = logging.getLogger(__name__)


class LatentExtractor(FeaturesExtractor):

    def __init__(self, model_name: str):
        FeaturesExtractor.__init__(self)
        self.__encoder = load_encoder(model_name)

        in_shape = self.__encoder.input_shape[1:3]   # drop batch dim
        out_shape = self.__encoder.output_shape[1:]  # drop batch dim
        
        self.__input_shape = in_shape
        self._set_n_features(int(np.prod(out_shape)))
        _logger.info("Loaded an encoder with %d features" % (self.n_features()))

    def features(self, img_data, skip_normalize=True):
        if not np.all(np.equal(img_data.shape[1:], self.__input_shape)):
            img_data = [resize(img, self.__input_shape, mode='reflect').astype('float32') for img in img_data]
        img_data = np.array([img/img.max() for img in img_data]).astype('float32')
        return self.__encoder.predict(img_data)

    def get_encoder(self):
        return self.__encoder
