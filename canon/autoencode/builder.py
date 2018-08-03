import keras
import keras.layers as L
import numpy as np

CONV_4 = "conv_4"


def compile_autoencoder(encoder, decoder):
    IMAGE_SHAPE =(128, 128)
    inp = L.Input(IMAGE_SHAPE)
    code = encoder(inp)
    reconstruction = decoder(code)

    autoencoder = keras.models.Model(inputs=inp, outputs=reconstruction)
    autoencoder.compile(optimizer="adamax", loss='binary_crossentropy')

    return autoencoder


def build(model_name: str, latent_features: int):
    if model_name == CONV_4:
        return conv_4((128, 128), latent_features)
    else:
        raise Exception("Unknown model name " + model_name)


def conv_4(img_shape, n=4):
    H, W = img_shape

    # encoder
    encoder = keras.models.Sequential()
    encoder.add(L.InputLayer(img_shape))
    encoder.add(L.Reshape((H, W, 1)))
    encoder.add(L.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
    encoder.add(L.MaxPooling2D(pool_size=(2, 2), padding='same'))
    encoder.add(L.Dropout(0.2))
    encoder.add(L.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    encoder.add(L.MaxPooling2D(pool_size=(2, 2), padding='same'))
    encoder.add(L.Dropout(0.2))
    encoder.add(L.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    encoder.add(L.MaxPooling2D(pool_size=(2, 2), padding='same'))
    encoder.add(L.Dropout(0.2))
    encoder.add(L.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    encoder.add(L.MaxPooling2D(pool_size=(2, 2), padding='same'))
    encoder.add(L.Dropout(0.2))
    encoder.add(L.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    encoder.add(L.MaxPooling2D(pool_size=(2, 2), padding='same'))
    encoder.add(L.Dropout(0.5))
    encoded_img_size = encoder.layers[-1].output_shape[1:]
    encoder.add(L.Flatten())
    encoder.add(L.Dense(n, activation='relu'))

    flatten_size = np.prod(encoded_img_size)

    # decoder
    decoder = keras.models.Sequential()
    decoder.add(L.InputLayer((n,)))
    decoder.add(L.Dropout(0.5))
    decoder.add(L.Dense(flatten_size, activation='relu'))
    decoder.add(L.Reshape(encoded_img_size))
    decoder.add(L.Dropout(0.2))
    decoder.add(L.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=2, activation='relu', padding='same'))
    decoder.add(L.Dropout(0.2))
    decoder.add(L.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, activation='relu', padding='same'))
    decoder.add(L.Dropout(0.2))
    decoder.add(L.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=2, activation='relu', padding='same'))
    decoder.add(L.Dropout(0.2))
    decoder.add(L.Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=2, activation='relu', padding='same'))
    decoder.add(L.Dropout(0.2))
    decoder.add(L.Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=2, activation='sigmoid', padding='same'))
    output_img_size = decoder.layers[-1].output_shape[1:-1]
    decoder.add(L.Reshape(output_img_size))

    return encoder, decoder


if __name__ == "__main__":
    from canon.autoencode import reset_tf_session
    s = reset_tf_session()
    encoder, decoder = build(CONV_4, 32)
    encoder.summary()
    decoder.summary()
