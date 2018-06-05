import logging
import numpy as np
from canon.autoencode.models import load_encoder
from canon.pattern.feature_extractor import FeaturesExtractor

_logger = logging.getLogger(__name__)


class LatentExtractor(FeaturesExtractor):

    def __init__(self, model_name: str):
        FeaturesExtractor.__init__(self)
        self.__encoder = load_encoder(model_name)
        self.__input_shape = self.__encoder.layers[0].input_shape[1:]
        self._set_n_features(int(np.prod(self.__encoder.layers[-1].output_shape[1:])))
        _logger.info("Loaded an encoder with %d features" % (self.n_features()))

    def features(self, img_data, skip_normalize=True):
        img_data = img_data.astype("float32") / 255.
        return self.__encoder.predict(img_data)

    def get_encoder(self):
        return self.__encoder
