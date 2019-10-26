import keras
import keras.backend as K
import keras.layers as L
from keras import losses
from keras import optimizers
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


def intensity_weight(y_true):
    return K.log(20000 * y_true + 1) + 1


def weighted_mse_loss(y_true, y_pred):
    weights, normalizer = intensity_weight(y_true)
    return K.mean(weights * K.square(y_true - y_pred))


def weighted_bce_loss(y_true, y_pred):
    weights, normalizer = intensity_weight(y_true)
    bce = - y_true * K.log(y_pred + K.epsilon()) - (1 - y_true) * K.log(1 - y_pred + K.epsilon())
    return K.mean(weights * bce)


def compile_autoencoder(encoder, decoder):
    IMAGE_SHAPE =(128, 128)
    inp = L.Input(IMAGE_SHAPE)
    code = encoder(inp)
    reconstruction = decoder(code)

    autoencoder = keras.models.Model(inputs=inp, outputs=reconstruction)
    autoencoder.compile(optimizer=optimizers.Adam(), loss=losses.binary_crossentropy, metrics=['mse'])

    return autoencoder


def build(model_name: str, latent_features: int):
    if model_name == LINEAR:
        return linear((128, 128), latent_features)

    if model_name == CONV_2:
        return conv((128, 128), latent_features, conv=2)
    if model_name == CONV_3:
        return conv((128, 128), latent_features, conv=3)
    if model_name == CONV_4:
        return conv((128, 128), latent_features, conv=4)
    if model_name == CONV_5:
        return conv((128, 128), latent_features, conv=5)
    if model_name == CONV_6:
        return conv((128, 128), latent_features, conv=6)
    if model_name == CONV_7:
        return conv((128, 128), latent_features, conv=7)

    if model_name == CONV_2_DENS_1:
        return conv((128, 128), latent_features, conv=2, dens=[1024])
    if model_name == CONV_3_DENS_1:
        return conv((128, 128), latent_features, conv=3, dens=[1024])
    if model_name == CONV_4_DENS_1:
        return conv((128, 128), latent_features, conv=4, dens=[1024])
    if model_name == CONV_5_DENS_1:
        return conv((128, 128), latent_features, conv=5, dens=[1024])

    if model_name == CONV2_2:
        return conv2((128, 128), latent_features, conv=2)
    if model_name == CONV2_3:
        return conv2((128, 128), latent_features, conv=3)

    if model_name == CONV2_5_DENS_1:
        return conv2((128, 128), latent_features, conv=5, dens=[1024])

    else:
        raise Exception("Unknown model name " + model_name)


def linear(img_shape, n=256):
    H, W = img_shape

    encoder = keras.models.Sequential()
    encoder.add(L.InputLayer(img_shape))
    encoder.add(L.Reshape((H*W,)))
    encoder.add(L.Dropout(0.5))
    encoder.add(L.Dense(n))

    decoder = keras.models.Sequential()
    decoder.add(L.InputLayer((n,)))
    decoder.add(L.Dropout(0.5))
    decoder.add(L.Dense(H*W))
    decoder.add(L.Reshape(img_shape))
    return encoder, decoder


def conv(img_shape, n=4, conv=4, dens=[]):
    H, W = img_shape

    # encoder
    encoder = keras.models.Sequential()
    encoder.add(L.InputLayer(img_shape))
    encoder.add(L.Reshape((H, W, 1)))
    for c in range(conv):
        filters = 16 * (2 ** c)
        encoder.add(L.Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same'))
        encoder.add(L.MaxPooling2D(pool_size=(2, 2), padding='same'))
        dropout = 0.2 if c == conv - 1 else 0.2
        encoder.add(L.Dropout(dropout))
    encoded_img_size = encoder.layers[-1].output_shape[1:]
    encoder.add(L.Flatten())
    for d in dens:
        encoder.add(L.Dense(d, activation='relu'))
        encoder.add(L.Dropout(0.2))
    encoder.add(L.Dense(n, activation='relu'))

    flatten_size = np.prod(encoded_img_size)

    # decoder
    decoder = keras.models.Sequential()
    decoder.add(L.InputLayer((n,)))
    for d in reversed(dens):
        decoder.add(L.Dropout(0.2))
        decoder.add(L.Dense(d, activation='relu'))
    decoder.add(L.Dropout(0.2))
    decoder.add(L.Dense(flatten_size, activation='relu'))
    decoder.add(L.Reshape(encoded_img_size))
    for c in range(conv - 1):
        filters = 4 * 2 ** (conv - c)
        decoder.add(L.Dropout(0.2))
        decoder.add(L.Conv2DTranspose(filters=filters, kernel_size=(3, 3), strides=2, activation='relu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=2, activation='sigmoid', padding='same'))
    output_img_size = decoder.layers[-1].output_shape[1:-1]
    decoder.add(L.Reshape(output_img_size))

    return encoder, decoder


def conv2(img_shape, n=4, conv=4, dens=[]):
    H, W = img_shape

    # encoder
    encoder = keras.models.Sequential()
    encoder.add(L.InputLayer(img_shape))
    encoder.add(L.Reshape((H, W, 1)))
    for c in range(conv):
        filters = 16 * (2 ** c)
        encoder.add(L.Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same'))
        encoder.add(L.Dropout(0.2))
        encoder.add(L.Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same'))
        encoder.add(L.MaxPooling2D(pool_size=(2, 2), padding='same'))
        dropout = 0.5 if c == conv - 1 else 0.2
        encoder.add(L.Dropout(dropout))
    encoded_img_size = encoder.layers[-1].output_shape[1:]
    encoder.add(L.Flatten())
    for d in dens:
        encoder.add(L.Dense(d, activation='relu'))
        encoder.add(L.Dropout(0.5))
    encoder.add(L.Dense(n, activation='relu'))

    flatten_size = np.prod(encoded_img_size)

    # decoder
    decoder = keras.models.Sequential()
    decoder.add(L.InputLayer((n,)))
    for d in reversed(dens):
        decoder.add(L.Dropout(0.5))
        decoder.add(L.Dense(d, activation='relu'))
    decoder.add(L.Dropout(0.5))
    decoder.add(L.Dense(flatten_size, activation='relu'))
    decoder.add(L.Reshape(encoded_img_size))
    filters = 0
    for c in range(conv - 1):
        filters = 2 ** (conv - c + 2)
        decoder.add(L.Dropout(0.2))
        decoder.add(
            L.Conv2DTranspose(filters=filters, kernel_size=(3, 3), strides=2, activation='relu', padding='same'))
        decoder.add(L.Dropout(0.2))
        decoder.add(
            L.Conv2DTranspose(filters=filters, kernel_size=(3, 3), strides=1, activation='relu', padding='same'))
    if filters > 0:
        decoder.add(L.Dropout(0.2))
        decoder.add(
            L.Conv2DTranspose(filters=filters, kernel_size=(3, 3), strides=2, activation='relu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=1, activation='sigmoid', padding='same'))
    output_img_size = decoder.layers[-1].output_shape[1:-1]
    decoder.add(L.Reshape(output_img_size))

    return encoder, decoder


if __name__ == "__main__":
    from canon.autoencode import reset_tf_session
    s = reset_tf_session()
    encoder, decoder = conv((128, 128), n=256, conv=3)

    encoder.summary()
    decoder.summary()
