import os
import logging
import numpy as np
from skimage.io import imread
from keras.utils import Sequence
from timeit import default_timer as timer


_logger = logging.getLogger(__name__)


class ImageDataFeeder(Sequence):

    EPOCH_SIZE = 100

    def __init__(self, batch_size: int, test_size: int, img_dir: str):
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.file_names = [filename for filename in os.listdir(img_dir)]
        self.test_files = self.file_names[:test_size]
        self.train_files = self.file_names[test_size:]

        self.__generate_epoch()

        _logger.info("Initialized a WhiteSequence of %d images" % (len(self.file_names)))

    def __len__(self):
        # return int(np.ceil(len(self.train_files) / float(self.batch_size)))
        return ImageDataFeeder.EPOCH_SIZE

    def __getitem__(self, idx):
        t0 = timer()
        batch_start = idx * self.batch_size
        batch_end = min((idx + 1) * self.batch_size, len(self.train_files))
        # batch_file_names = self.train_files[batch_start:batch_end]
        # X_batch = self.__to_data_matrix(batch_file_names)
        X_batch = self.X_epoch[batch_start:batch_end]
        # _logger.info("Read a batch of %d images (%g sec)" % (len(X_batch), timer() - t0))
        return X_batch, X_batch

    def __to_data_matrix(self, file_names):
        data = np.array([imread(os.path.join(self.img_dir, f)) for f in file_names])
        data = data.astype('float32') / 255.
        return data

    def __generate_epoch(self):
        epoch_files = np.random.choice(self.train_files, ImageDataFeeder.EPOCH_SIZE * self.batch_size)
        self.X_epoch = self.__to_data_matrix(epoch_files)

    # noinspection PyNoneFunctionAssignment
    def on_epoch_end(self):
        self.__generate_epoch()
        # self.file_names = np.random.shuffle(self.train_files)

    def get_test_set(self):
        return self.__to_data_matrix(self.test_files)

    def get_training_set(self):
        return self.__to_data_matrix(self.train_files[:min(ImageDataFeeder.EPOCH_SIZE * self.batch_size, len(self.train_files))])

