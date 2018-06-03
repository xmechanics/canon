import os
import logging
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from keras.models import model_from_json


from canon.pattern.feature_extractor import FeaturesExtractor
from canon.img.tiff_reader import TiffReader

_logger = logging.getLogger(__name__)


class LatentExtractor(FeaturesExtractor):

    def __init__(self, model_name: str):
        FeaturesExtractor.__init__(self)
        model_dir = self.__get_model_dir(model_name)
        with open(os.path.join(model_dir, 'encoder.json'), 'r') as json_file:
            loaded_encoder = model_from_json(json_file.read())
            loaded_encoder.load_weights(os.path.join(model_dir, 'encoder.h5'))
        self.__encoder = loaded_encoder
        self.__input_shape = self.__encoder.layers[0].input_shape[1:]
        self._set_n_features(int(np.prod(self.__encoder.layers[-1].output_shape[1:])))
        _logger.info("Loaded an encoder with %d features" % (self.n_features()))

    def features(self, tiff_files, skip_normalize=True):
        reader = TiffReader('pilatus')
        imgs = []
        for tiff in tiff_files:
            if not skip_normalize:
                reader.loadtiff(tiff)
                reader.remove_background()
                reader.normalize()
                img = reader.image()
                img = img[:981, :]
                img = resize(img, self.__input_shape, mode='reflect')
            else:
                img = imread(tiff).astype("float32")
            img = img / 255.
            imgs.append(img)
            # if len(imgs) % 100 == 0:
            #     _logger.info("Normalized %d tiff files" % len(imgs))
        imgs = np.array(imgs)
        return self.__encoder.predict(imgs)

    def get_encoder(self):
        return self.__encoder

    @staticmethod
    def __get_model_dir(model_name):
        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        return os.path.join(project_dir, "data", "models", model_name)


if __name__ == "__main__":
    from canon.common.init import init_mpi_logging
    init_mpi_logging("logging.yaml")
    extractor = LatentExtractor("AE_20180602")
    images = [
        "../../scripts/img/CuAlNi_mart2/CuAlNi_mart2_00001.tif",
        "../../scripts/img/CuAlNi_mart2/CuAlNi_mart2_00002.tif"
    ]
    extractor.features(images)


