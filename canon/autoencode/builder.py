import keras
import keras.backend as K
import keras.layers as L
from keras import losses, optimizers
import numpy as np

LINEAR = "linear"
CONV_2 = "conv_2"
CONV_3 = "conv_3"
CONV_4 = "conv_4"
CONV_5 = "conv_5"
CONV_6 = "conv_6"
CONV_7 = "conv_7"

CONV_2_DENS_1 = "conv_2_dens_1"
CONV_3_DENS_1 = "conv_3_dens_1"
CONV_4_DENS_1 = "conv_4_dens_1"
CONV_5_DENS_1 = "conv_5_dens_1"

CONV2_2 = "conv2_2"
CONV2_3 = "conv2_3"
CONV2_5_DENS_1 = "conv2_5_dens1"


# ---------------------------------------------------------------------------
#  Weighting & custom losses
# ---------------------------------------------------------------------------

def _intensity_weight(y_true):
    """Pixel‑wise weight ~log(intensity)."""
    return K.log(2e4 * y_true + 1.) + 1.


def weighted_mse_loss(y_true, y_pred):
    w = _intensity_weight(y_true)
    return K.mean(w * K.square(y_true - y_pred))


def weighted_bce_loss(y_true, y_pred):
    w = _intensity_weight(y_true)
    bce = -y_true * K.log(y_pred + K.epsilon()) - (1. - y_true) * K.log(1. - y_pred + K.epsilon())
    return K.mean(w * bce)


# ---------------------------------------------------------------------------
#  Public helpers
# ---------------------------------------------------------------------------

def compile_autoencoder(encoder, decoder):
    IMAGE_SHAPE = (128, 128)
    inp = L.Input(IMAGE_SHAPE)
    out = decoder(encoder(inp))
    auto = keras.Model(inp, out, name="autoencoder")
    auto.compile(optimizer=optimizers.Adam(), loss=losses.binary_crossentropy, metrics=["mse"])
    return auto


def build(model_name: str, latent_features: int):
    if model_name == LINEAR:
        return linear((128, 128), latent_features)

    if model_name in {CONV_2, CONV_3, CONV_4, CONV_5, CONV_6, CONV_7}:
        c = int(model_name.split("_")[-1])  # get 2/3/4… from "conv_4"
        return conv((128, 128), latent_features, conv=c)

    if model_name.endswith("dens_1"):
        base, _, _ = model_name.partition("_dens_1")
        if base.startswith("conv2_"):
            c = int(base.split("_")[-1])
            return conv2((128, 128), latent_features, conv=c, dens=[1024])
        else:
            c = int(base.split("_")[-1])
            return conv((128, 128), latent_features, conv=c, dens=[1024])

    if model_name in {CONV2_2, CONV2_3}:
        c = int(model_name.split("_")[-1])
        return conv2((128, 128), latent_features, conv=c)

    if model_name == CONV2_5_DENS_1:
        return conv2((128, 128), latent_features, conv=5, dens=[1024])

    raise ValueError(f"Unknown model name {model_name}")


# ---------------------------------------------------------------------------
#  Builders
# ---------------------------------------------------------------------------

def linear(img_shape, n=256):
    H, W = img_shape
    encoder = keras.Sequential([
        L.InputLayer(img_shape),
        L.Reshape((H * W,)),
        L.Dropout(0.5),
        L.Dense(n, activation="relu"),
    ], name="linear_encoder")

    decoder = keras.Sequential([
        L.InputLayer((n,)),
        L.Dropout(0.5),
        L.Dense(H * W, activation="sigmoid"),
        L.Reshape(img_shape),
    ], name="linear_decoder")

    return encoder, decoder


def _encoded_shape(img_shape, conv_blocks):
    """Compute (H, W, C) of the last conv layer programmatically."""
    H, W = img_shape
    h_enc = H // (2 ** conv_blocks)
    w_enc = W // (2 ** conv_blocks)
    c_enc = 16 * (2 ** (conv_blocks - 1))
    return h_enc, w_enc, c_enc


def conv(img_shape, n=4, conv=4, dens=None):
    dens = dens or []
    H, W = img_shape

    # ---------------- Encoder ----------------
    encoder_layers = [L.InputLayer(img_shape), L.Reshape((H, W, 1))]
    for c in range(conv):
        filters = 16 * (2 ** c)
        encoder_layers += [
            L.Conv2D(filters, 3, activation="relu", padding="same"),
            L.MaxPooling2D(2, padding="same"),
            L.Dropout(0.2),
        ]
    encoder_layers.append(L.Flatten())
    for units in dens:
        encoder_layers += [L.Dense(units, activation="relu"), L.Dropout(0.2)]
    encoder_layers.append(L.Dense(n, activation="relu"))

    encoder = keras.Sequential(encoder_layers, name="conv_encoder")

    # Analytically derive encoded feature‑map size
    encoded_img_size = _encoded_shape(img_shape, conv)
    flatten_size = int(np.prod(encoded_img_size))

    # ---------------- Decoder ----------------
    decoder_layers = [L.InputLayer((n,))]
    for units in reversed(dens):
        decoder_layers += [L.Dropout(0.2), L.Dense(units, activation="relu")]
    decoder_layers += [
        L.Dropout(0.2),
        L.Dense(flatten_size, activation="relu"),
        L.Reshape(encoded_img_size),
    ]
    for c in range(conv - 1):
        filters = 4 * 2 ** (conv - c)
        decoder_layers += [
            L.Dropout(0.2),
            L.Conv2DTranspose(filters, 3, strides=2, activation="relu", padding="same"),
        ]
    decoder_layers += [
        L.Conv2DTranspose(1, 3, strides=2, activation="sigmoid", padding="same"),
        L.Reshape(img_shape),
    ]

    decoder = keras.Sequential(decoder_layers, name="conv_decoder")

    return encoder, decoder


def conv2(img_shape, n=4, conv=4, dens=None):
    dens = dens or []
    H, W = img_shape

    encoder_layers = [L.InputLayer(img_shape), L.Reshape((H, W, 1))]
    for c in range(conv):
        filters = 16 * (2 ** c)
        encoder_layers += [
            L.Conv2D(filters, 3, activation="relu", padding="same"),
            L.Dropout(0.2),
            L.Conv2D(filters, 3, activation="relu", padding="same"),
            L.MaxPooling2D(2, padding="same"),
            L.Dropout(0.5 if c == conv - 1 else 0.2),
        ]
    encoder_layers.append(L.Flatten())
    for units in dens:
        encoder_layers += [L.Dense(units, activation="relu"), L.Dropout(0.5)]
    encoder_layers.append(L.Dense(n, activation="relu"))

    encoder = keras.Sequential(encoder_layers, name="conv2_encoder")

    encoded_img_size = _encoded_shape(img_shape, conv)
    flatten_size = int(np.prod(encoded_img_size))

    decoder_layers = [L.InputLayer((n,))]
    for units in reversed(dens):
        decoder_layers += [L.Dropout(0.5), L.Dense(units, activation="relu")]
    decoder_layers += [
        L.Dropout(0.5),
        L.Dense(flatten_size, activation="relu"),
        L.Reshape(encoded_img_size),
    ]

    filters = 0
    for c in range(conv - 1):
        filters = 2 ** (conv - c + 2)
        decoder_layers += [
            L.Dropout(0.2),
            L.Conv2DTranspose(filters, 3, strides=2, activation="relu", padding="same"),
            L.Dropout(0.2),
            L.Conv2DTranspose(filters, 3, strides=1, activation="relu", padding="same"),
        ]
    if filters > 0:
        decoder_layers += [
            L.Dropout(0.2),
            L.Conv2DTranspose(filters, 3, strides=2, activation="relu", padding="same"),
        ]
    decoder_layers += [
        L.Conv2DTranspose(1, 3, strides=1, activation="sigmoid", padding="same"),
        L.Reshape(img_shape),
    ]

    decoder = keras.Sequential(decoder_layers, name="conv2_decoder")

    return encoder, decoder


# ---------------------------------------------------------------------------
#  Simple self‑test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    enc, dec = conv((128, 128), n=256, conv=4)
    auto = compile_autoencoder(enc, dec)
    enc.summary()  # prints shapes
    dec.summary()  # prints shapes
