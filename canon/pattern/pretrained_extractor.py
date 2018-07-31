import logging
import numpy as np
from skimage.transform import resize
from keras.models import Model
from keras.layers import Flatten
from keras.applications import VGG16, VGG19, InceptionV3

from canon.pattern.feature_extractor import FeaturesExtractor

_logger = logging.getLogger(__name__)


class PretrainedExtractor(FeaturesExtractor):

    def __init__(self, app_name: str):
        FeaturesExtractor.__init__(self)
        self.__model = self.__get_model(app_name)
        self.__input_shape = self.__model.layers[0].input_shape[1:-1]
        self._set_n_features(int(np.prod(self.__model.layers[-1].output_shape[1:])))
        _logger.info("Loaded an extractor with %s input size and %d features" % (self.__input_shape, self.n_features()))

    def features(self, img_data, skip_normalize=True):
        if not np.all(np.equal(img_data.shape[1:], self.__input_shape)):
            img_data = [resize(img, self.__input_shape, mode='reflect').astype('float32') for img in img_data]
        bw_data = np.array([255. * img/img.max() for img in img_data]).astype('int32')
        rgb_data = np.zeros((bw_data.shape[0], bw_data.shape[1], bw_data.shape[2], 3))
        rgb_data[:, :, :, 0] = bw_data.copy()
        rgb_data[:, :, :, 1] = bw_data.copy()
        rgb_data[:, :, :, 2] = bw_data.copy()
        return self.__model.predict(rgb_data)

    @staticmethod
    def __get_model(app_name):
        input_size = PretrainedExtractor.__get_input_size(app_name)
        app = None
        if app_name.lower() == "vgg16":
            app = VGG16(weights='imagenet', include_top=False, input_shape=input_size)
        elif app_name.lower() == "vgg19":
            app = VGG19(weights='imagenet', include_top=False, input_shape=input_size)
        elif app_name.lower() == "inceptionv3":
            app = InceptionV3(weights='imagenet', include_top=False, input_shape=input_size)
        else:
            raise ValueError("Unknown application name %s" % app_name)
        predictions = Flatten()(app.output)
        return Model(inputs=app.input, outputs=predictions)

    @staticmethod
    def __get_input_size(app_name):
        return (224, 224, 3)


if __name__ == "__main__":
    from canon.common.init import init_logging
    init_logging()

    # model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # model.summary()
    extractor = PretrainedExtractor('vgg19')

