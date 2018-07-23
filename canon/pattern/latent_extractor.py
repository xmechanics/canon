import logging
import os
import numpy as np
from skimage.transform import resize
from keras.models import model_from_json

from canon.pattern.feature_extractor import FeaturesExtractor

_logger = logging.getLogger(__name__)


def load_encoder(model_name: str):
    model_dir = __get_model_dir(model_name)
    with open(os.path.join(model_dir, 'encoder.json'), 'r') as json_file:
        loaded_encoder = model_from_json(json_file.read())
        loaded_encoder.load_weights(os.path.join(model_dir, 'encoder.h5'))
    return loaded_encoder


def load_decoder(model_name: str):
    model_dir = __get_model_dir(model_name)
    with open(os.path.join(model_dir, 'decoder.json'), 'r') as json_file:
        loaded_decoder = model_from_json(json_file.read())
        loaded_decoder.load_weights(os.path.join(model_dir, 'decoder.h5'))
    return loaded_decoder


def __get_model_dir(model_name):
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    return os.path.join(project_dir, "data", "models", model_name)


class LatentExtractor(FeaturesExtractor):

    def __init__(self, model_name: str):
        FeaturesExtractor.__init__(self)
        self.__encoder = load_encoder(model_name)
        self.__input_shape = self.__encoder.layers[0].input_shape[1:]
        self._set_n_features(int(np.prod(self.__encoder.layers[-1].output_shape[1:])))
        _logger.info("Loaded an encoder with %d features" % (self.n_features()))

    def features(self, img_data, skip_normalize=True):
        if not np.all(np.equal(img_data.shape[1:], self.__input_shape)):
            img_data = [resize(img, self.__input_shape, mode='reflect') for img in img_data]
        img_data = img_data.astype('float32')
        img_data = np.array([img > 0.2 * img.max() for img in img_data])
        return self.__encoder.predict(img_data)

    def get_encoder(self):
        return self.__encoder
