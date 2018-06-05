import keras
import keras.layers as L

from canon.autoencode.models import AE_128_to_256, AE_128_to_64


def compile_autoencoder(encoder, decoder):
    IMAGE_SHAPE =(128, 128)
    inp = L.Input(IMAGE_SHAPE)
    code = encoder(inp)
    reconstruction = decoder(code)

    autoencoder = keras.models.Model(inputs=inp, outputs=reconstruction)
    autoencoder.compile(optimizer="adamax", loss='mse')
    return autoencoder


def build(model_name: str):
    if model_name == AE_128_to_256:
        return build_deep_autoencoder((128, 128), 256)
    elif model_name == AE_128_to_64:
        return build_to_64((128, 128))
    else:
        raise Exception("Unknown model name " + model_name)


def build_to_64(img_shape):
    H, W = img_shape
    code_size = 64

    # encoder
    encoder = keras.models.Sequential()
    encoder.add(L.InputLayer(img_shape))
    encoder.add(L.Reshape((H, W, 1)))
    encoder.add(L.Conv2D(8, kernel_size=(3, 3), activation='elu', padding='same'))
    encoder.add(L.MaxPooling2D(pool_size=(2, 2), padding='same'))
    encoder.add(L.Conv2D(16, kernel_size=(3, 3), activation='elu', padding='same'))
    encoder.add(L.MaxPooling2D(pool_size=(2, 2), padding='same'))
    encoder.add(L.Conv2D(32, kernel_size=(3, 3), activation='elu', padding='same'))
    encoder.add(L.MaxPooling2D(pool_size=(2, 2), padding='same'))
    encoder.add(L.Flatten())
    encoder.add(L.Dense(code_size, activation='elu'))

    flatten_size = encoder.layers[-2].output_shape[1]
    encoded_img_size = encoder.layers[-3].output_shape[1:]

    # decoder
    decoder = keras.models.Sequential()
    decoder.add(L.InputLayer((code_size,)))
    decoder.add(L.Dense(flatten_size, activation='elu'))
    decoder.add(L.Reshape(encoded_img_size))
    decoder.add(L.Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=8, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=2, activation='sigmoid', padding='same'))
    output_img_size = decoder.layers[-1].output_shape[1:-1]
    decoder.add(L.Reshape(output_img_size))

    return encoder, decoder


def build_deep_autoencoder(img_shape, code_size):
    H, W = img_shape

    # encoder
    encoder = keras.models.Sequential()
    encoder.add(L.InputLayer(img_shape))
    encoder.add(L.Reshape((H, W, 1)))
    encoder.add(L.Conv2D(8, kernel_size=(3, 3), activation='elu', padding='same'))
    encoder.add(L.MaxPooling2D(pool_size=(2, 2), padding='same'))
    encoder.add(L.Conv2D(16, kernel_size=(3, 3), activation='elu', padding='same'))
    encoder.add(L.MaxPooling2D(pool_size=(2, 2), padding='same'))
    encoder.add(L.Conv2D(32, kernel_size=(3, 3), activation='elu', padding='same'))
    encoder.add(L.MaxPooling2D(pool_size=(2, 2), padding='same'))
    encoder.add(L.Conv2D(64, kernel_size=(3, 3), activation='elu', padding='same'))
    encoder.add(L.MaxPooling2D(pool_size=(2, 2), padding='same'))
    encoder.add(L.Flatten())
    encoder.add(L.Dense(1024, activation='elu'))
    encoder.add(L.Dense(code_size, activation='elu'))

    flatten_size = encoder.layers[-3].output_shape[1]
    encoded_img_size = encoder.layers[-4].output_shape[1:]

    # decoder
    decoder = keras.models.Sequential()
    decoder.add(L.InputLayer((code_size,)))
    decoder.add(L.Dense(1024, activation='elu'))
    decoder.add(L.Dense(flatten_size, activation='elu'))
    decoder.add(L.Reshape(encoded_img_size))
    decoder.add(L.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=8, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=2, activation='sigmoid', padding='same'))
    output_img_size = decoder.layers[-1].output_shape[1:-1]
    decoder.add(L.Reshape(output_img_size))

    return encoder, decoder


if __name__ == "__main__":
    from canon.autoencode import reset_tf_session

    IMAGE_SHAPE = (128, 128)
    CODE_SIZE = 256

    s = reset_tf_session()
    encoder, decoder = build(AE_128_to_64)
    encoder.summary()
    decoder.summary()