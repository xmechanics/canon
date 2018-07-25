import os
import logging
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from keras.utils import Sequence
from joblib import Parallel, delayed

_logger = logging.getLogger(__name__)


def load_img(f, shape):
    img = resize(imread(f), shape, mode='reflect').astype('float32')
    return img / img.max()


class ImageDataFeeder(Sequence):

    def __init__(self, img_shape, batch_size: int, training_dir: str, test_dir: str):
        self.img_shape = img_shape;
        self.training_dir = training_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.test_files = [filename for filename in os.listdir(test_dir)]
        self.training_files = [filename for filename in os.listdir(training_dir) if filename not in set(self.test_files)]
        _logger.info("Initialized a WhiteSequence of %d training images and %d test images" %
                     (len(self.training_files), len(self.test_files)))
        self.epoch_size = int(np.ceil(len(self.training_files) / float(self.batch_size)))

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        batch_start = idx * self.batch_size
        batch_end = min((idx + 1) * self.batch_size, len(self.training_files))
        batch_file_names = self.training_files[batch_start:batch_end]
        X_batch = self.__to_data_matrix(self.training_dir, batch_file_names)
        return X_batch, X_batch

    def __to_data_matrix(self, dir_name, file_names):
        _logger.info("Reading %d image files into array" % len(file_names))

        with Parallel(n_jobs=-1, verbose=11) as parallel:
            data = parallel(delayed(load_img)((os.path.join(dir_name, f)), self.img_shape) for f in file_names)
            data = np.array(data)
            _logger.info("Loaded a data of shape {}: max={}, min={}".format(data.shape, data.max(), data.min()))
        return data

    def __generate_epoch(self):
        epoch_files = np.random.choice(self.training_files, self.epoch_size * self.batch_size)
        self.X_epoch = self.__to_data_matrix(self.training_dir,  epoch_files)

    # noinspection PyNoneFunctionAssignment
    def on_epoch_end(self):
        self.training_files = np.random.shuffle(self.training_files)

    def get_test_set(self):
        return self.__to_data_matrix(self.test_dir, self.test_files)

    def get_training_set(self):
        return self.__to_data_matrix(self.training_dir, self.training_files)
