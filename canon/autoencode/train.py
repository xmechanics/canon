import os
import time
import keras
from keras.callbacks import TensorBoard

from canon.autoencode.feeder import ImageDataFeeder
from canon.autoencode.builder import compile_autoencoder, build


class ModelSaveCallback(keras.callbacks.Callback):
    def __init__(self, file_name):
        super(ModelSaveCallback, self).__init__()
        self.file_name = file_name

    def on_epoch_end(self, epoch, logs=None):
        model_filename = self.file_name.format(epoch)
        keras.models.save_model(self.model, model_filename)
        print("Model saved in {}".format(model_filename))


def train(model_name, feed_dir, epochs=10000, initial_epoch=0, checkpoint=None):
    if checkpoint is not None:
        autoencoder = keras.models.load_model(checkpoint)
        encoder = autoencoder.layers[1]
        decoder = autoencoder.layers[2]
    else:
        encoder, decoder = build(model_name)
        autoencoder = compile_autoencoder(encoder, decoder)

    feeder = ImageDataFeeder(batch_size=30, test_size=500, img_dir=feed_dir)
    X_test = feeder.get_test_set()
    X_train = feeder.get_training_set()
    run_number = time.time()
    checkpoint_dir = "checkpoints/{}/{}".format(model_name, run_number)
    model_dir = "models/{}/{}".format(model_name, run_number)
    os.makedirs(checkpoint_dir)
    os.makedirs(model_dir)

    callbacks = [TensorBoard(log_dir="logs/{}".format(run_number)),
                 ModelSaveCallback(checkpoint_dir + "/autoencoder.{0:03d}.hdf5")]

    autoencoder.fit(X_train, X_train,
                    epochs=epochs,
                    shuffle=True,
                    validation_data=[X_test, X_test],
                    callbacks=callbacks,
                    verbose=1,
                    initial_epoch=initial_epoch)

    #     autoencoder.fit_generator(feeder, epochs=20,
    #                     validation_data=[X_test, X_test],
    #                     callbacks=[TensorBoard(log_dir="./logs/{}".format(time.time()))],
    #                     verbose=1,
    #                     initial_epoch=last_finished_epoch or 0)


