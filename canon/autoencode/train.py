import os
import time
import logging
import keras
from keras.callbacks import TensorBoard

from canon.autoencode.feeder import ImageDataFeeder
from canon.autoencode.builder import compile_autoencoder, build


_logger = logging.getLogger(__name__)


class ModelSaveCallback(keras.callbacks.Callback):
    def __init__(self, file_name):
        super(ModelSaveCallback, self).__init__()
        self.file_name = file_name

    def on_epoch_end(self, epoch, logs=None):
        checkpoint = self.file_name.format(epoch)
        keras.models.save_model(self.model, checkpoint)
        _logger.info("Saved a new checkpoint in {}".format(checkpoint))
        previous_checkpoint = self.file_name.format(epoch - 1)
        if os.path.exists(previous_checkpoint):
            os.remove(previous_checkpoint)
            _logger.info("Removed previous checkpoint {}".format(previous_checkpoint))


def train(architecture, run_number, training_dir, test_dir, epochs=10000, verbose=0, dryrun=False):
    checkpoint, initial_epoch = find_checkpoint(architecture, run_number)
    if checkpoint is not None:
        _logger.info("Found initial_epoch={} in checkpoint {}".format(initial_epoch, checkpoint))
        autoencoder = keras.models.load_model(checkpoint)
        # autoencoder.compile(optimizer="adamax", loss='binary_crossentropy')
        encoder = autoencoder.layers[1]
        decoder = autoencoder.layers[2]
    else:
        _logger.info("Did not find checkpoint, start from scratch")
        encoder, decoder = build(architecture)
        autoencoder = compile_autoencoder(encoder, decoder)
        initial_epoch = 0

    encoder.summary()
    decoder.summary()

    if dryrun:
        return

    batch_size = 500

    feeder = ImageDataFeeder(batch_size=batch_size, training_dir=training_dir, test_dir=test_dir)
    X_test = feeder.get_test_set()
    X_train = feeder.get_training_set()
    if run_number is None:
        run_number = architecture
    checkpoint_dir = "checkpoints/{}/{}".format(architecture, run_number)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    callbacks = [TensorBoard(log_dir="logs/{}".format(run_number)),
                 ModelSaveCallback(checkpoint_dir + "/autoencoder.{0:03d}.hdf5")]

    autoencoder.fit(X_train, X_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=[X_test, X_test],
                    callbacks=callbacks,
                    verbose=verbose,
                    initial_epoch=initial_epoch)

    #     autoencoder.fit_generator(feeder, epochs=20,
    #                     validation_data=[X_test, X_test],
    #                     callbacks=[TensorBoard(log_dir="./logs/{}".format(time.time()))],
    #                     verbose=1,
    #                     initial_epoch=last_finished_epoch or 0)


def find_checkpoint(architecture, run_number):
    checkpoint_dir = "checkpoints/{}/{}".format(architecture, run_number)
    if os.path.exists(checkpoint_dir):
        fns = os.listdir(checkpoint_dir)
        if len(fns) >= 1:
            fns.sort()
            latest = fns[-1]
            epoch = int(latest.split(".")[1])
            checkpoint = os.path.join(checkpoint_dir, latest)
            return checkpoint, epoch
    return None, None


