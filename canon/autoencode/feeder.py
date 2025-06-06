import os
import logging
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from keras.utils import Sequence
from joblib import Parallel, delayed
import multiprocessing

from .report import is_using_gpu

_logger = logging.getLogger(__name__)


def load_img(f, shape):
    img = resize(imread(f), shape, mode='reflect').astype('float32')
    return img / img.max()


class ImageDataFeeder(Sequence):

    def __init__(self, img_shape, batch_size: int, training_dir: str, test_dir: str, enrich=False):
        self.img_shape = img_shape
        self.training_dir = training_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.enrich = enrich
        self.test_files = [filename for filename in os.listdir(test_dir)]
        self.training_files = [filename for filename in os.listdir(training_dir) if
                               (not filename[0] == '.') and filename[-4:] == ".jpg"
                               and (filename not in set(self.test_files))]
        self.parallelism = -1
        np.random.shuffle(self.training_files)
        _logger.info("Initialized a WhiteSequence of %d training images and %d test images" %
                     (len(self.training_files), len(self.test_files)))
        self.epoch_size = int(np.ceil(len(self.training_files) / float(self.batch_size)))

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        batch_start = idx * self.batch_size
        batch_end = min((idx + 1) * self.batch_size, len(self.training_files))
        batch_file_names = self.training_files[batch_start:batch_end]
        X_batch = self.__to_data_matrix(self.training_dir, batch_file_names, parallel=False, enrich=self.enrich)
        return X_batch, X_batch

    def __to_data_matrix(self, dir_name, file_names, parallel=True, enrich=False):
        _logger.info("Reading %d image files into array" % len(file_names))

        if parallel:
            with Parallel(n_jobs=self.parallelism, verbose=11) as parallel:
                data = parallel(delayed(load_img)((os.path.join(dir_name, f)), self.img_shape) for f in file_names)
        else:
            data = [load_img(os.path.join(dir_name, f), self.img_shape) for f in file_names]
        
        # Convert to numpy array before enriching to ensure consistent shapes
        data = np.array(data)
        _logger.info("data shape: {}".format(data.shape))

        if enrich:
            _logger.info("Enriching %d image data" % len(data))

            # Check if data is 3D or 4D
            if data.ndim == 3:
                # (N, H, W) --> grayscale
                axis_h, axis_w = 1, 2
            elif data.ndim == 4:
                # (N, H, W, C) --> with channels
                axis_h, axis_w = 1, 2
            else:
                raise ValueError(f"Unexpected data shape: {data.shape}")

            # Flip along height (vertical flip)
            data_flip_h = np.flip(data, axis=axis_h)

            # Flip along width (horizontal flip)
            data_flip_w = np.flip(data, axis=axis_w)

            # Transpose H and W
            if data.ndim == 3:
                data_transpose = np.transpose(data, (0, 2, 1))  # (N, W, H)
            else:
                data_transpose = np.transpose(data, (0, 2, 1, 3))  # (N, W, H, C)

            # Transpose and flip vertically
            data_transpose_flip = np.flip(data_transpose, axis=axis_h)

            # Concatenate all augmented data
            data = np.concatenate([data, data_flip_h, data_flip_w, data_transpose, data_transpose_flip], axis=0)

            _logger.info("Enriched data shape: {}".format(data.shape))

        _logger.info("Loaded a data of shape {}: max={}, min={}".format(data.shape, data.max(), data.min()))
        return data

    def __generate_epoch(self):
        epoch_files = np.random.choice(self.training_files, self.epoch_size * self.batch_size)
        self.X_epoch = self.__to_data_matrix(self.training_dir,  epoch_files)

    # noinspection PyNoneFunctionAssignment
    def on_epoch_end(self):
        np.random.shuffle(self.training_files)

    def get_test_set(self):
        return self.__to_data_matrix(self.test_dir, self.test_files)

    def get_training_set(self):
        return self.__to_data_matrix(self.training_dir, self.training_files, enrich=self.enrich)
