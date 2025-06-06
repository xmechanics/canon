import os
import logging
import keras
from keras import backend as K
from keras import optimizers
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint

from canon.autoencode.feeder import ImageDataFeeder
from canon.autoencode.builder_v2 import compile_autoencoder, build

_logger = logging.getLogger(__name__)


class ModelSaveCallback(keras.callbacks.Callback):
    """Save a rolling checkpoint (autoencoder.<epoch>.keras)."""

    def __init__(self, file_name):
        super().__init__()
        self.file_name = file_name

    def on_epoch_end(self, epoch, logs=None):
        checkpoint = self.file_name.format(epoch)
        # Keras 3 prefers the new .keras format, but model.save() still works
        self.model.save(checkpoint)
        _logger.info("Saved checkpoint %s", checkpoint)
        prev = self.file_name.format(epoch - 1)
        if os.path.exists(prev):
            os.remove(prev)
            _logger.info("Removed previous checkpoint %s", prev)


def weighted_mse_loss(y_true, y_pred):
    weights = K.log(2000 * y_true + 1.0) + 1.0
    return K.mean(weights * K.square(y_true - y_pred))


def train(
    backbone: str,
    n_features: int,
    training_dir: str,
    test_dir: str,
    epochs: int = 100,
    verbose: int = 1,
    dryrun: bool = False,
    use_generator: bool = False,
):
    """Train auto‑encoder; resume from checkpoints if present."""

    checkpoint_path, initial_epoch = find_checkpoint(backbone, n_features)
    if checkpoint_path is not None:
        _logger.info("Resuming from %s (epoch %d)", checkpoint_path, initial_epoch)
        autoencoder = keras.saving.load_model(checkpoint_path)
        encoder, decoder = autoencoder.layers[1], autoencoder.layers[2]
        try:
            latent = autoencoder.get_layer("ae_latent").output
        except ValueError:
            # fallback for legacy models where the encoder was layer[1]
            legacy_encoder = autoencoder.layers[1]
            # assume layer[1] is a Model or Sequential
            encoder = legacy_encoder

        # 3) new 2D-style model: build an encoder from auto.input → latent
        encoder = keras.Model(inputs=autoencoder.input,
                        outputs=latent,
                        name=f"{backbone}_encoder")
        # 2) grab the nested decoder Model by name
        try:
            decoder_layer = autoencoder.get_layer(f"{backbone}_decoder")
        except ValueError:
            raise ValueError(
                f"Decoder not found. "
                f"Make sure your build() named it '{backbone}_decoder'."
            )

        # 3) If that layer is *already* a Model (keras.Model subclass), return it
        if isinstance(decoder_layer, keras.Model):
            decoder = decoder_layer
        else:
            # 4) Otherwise wrap its input→output into a Model
            inp  = decoder_layer.input   # should be the latent-vector Input
            out  = decoder_layer.output  # should be the (H,W) output
            decoder = keras.Model(inp, out, name=f"{backbone}_decoder")
    else:
        _logger.info("No checkpoint found — building new model")
        encoder, decoder = build(backbone, n_features)
        autoencoder = compile_autoencoder(encoder, decoder)
        initial_epoch = 0

    encoder.summary()
    decoder.summary()
    if dryrun:
        return

    # ------------------------------------------------------------------
    #  Callbacks & logging dirs
    # ------------------------------------------------------------------
    ckpt_dir = os.path.join("checkpoints", backbone.lower(), str(n_features))
    os.makedirs(ckpt_dir, exist_ok=True)

    callbacks = [
        TensorBoard(log_dir=os.path.join("logs", f"{backbone.lower()}_{n_features}"), write_graph=False),
        ModelCheckpoint(
            filepath=os.path.join(ckpt_dir, "autoencoder.{epoch:03d}.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        CSVLogger(os.path.join(ckpt_dir, "history.csv"), separator=",", append=False),
    ]

    batch_size = 100 if use_generator else 500

    img_shape = encoder.input_shape[1:3]  # safe with Keras 3
    feeder = ImageDataFeeder(img_shape, batch_size=batch_size, training_dir=training_dir, test_dir=test_dir, enrich=True)
    X_test = feeder.get_test_set()

    if use_generator:
        autoencoder.fit(
            feeder,
            epochs=epochs,
            validation_data=(X_test, X_test),
            callbacks=callbacks,
            verbose=verbose,
            initial_epoch=initial_epoch,
        )
    else:
        X_train = feeder.get_training_set()
        autoencoder.fit(
            X_train,
            X_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(X_test, X_test),
            callbacks=callbacks,
            verbose=verbose,
            initial_epoch=initial_epoch,
        )


# ---------------------------------------------------------------------------
#  Checkpoint helper
# ---------------------------------------------------------------------------

def find_checkpoint(backbone: str, n_features: int):
    ckpt_dir = os.path.join("checkpoints", backbone.lower(), str(n_features))
    if os.path.exists(ckpt_dir):
        files = [f for f in os.listdir(ckpt_dir) if f.endswith(".keras")]
        if len(files) >= 1:
            files.sort()
            latest = files[-1]
            epoch = int(latest.split(".")[1])
            return os.path.join(ckpt_dir, latest), epoch
    return None, None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train(
        architecture="conv_3",
        n_features=256,
        training_dir="./data/train",
        test_dir="./data/test",
        epochs=5,
        verbose=1,
        dryrun=True,
    )
