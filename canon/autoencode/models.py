import os
import keras
import logging

__logger = logging.getLogger(__name__)

def load_autoencoder(model_name: str):
    model_dir = __get_model_dir(model_name)
    # Check if Keras model file exists for direct loading
    keras_model_path = os.path.join(model_dir, 'autoencoder.keras')
    if os.path.exists(keras_model_path):
        model = keras.models.load_model(keras_model_path)
        return model.layers[1], model.layers[2]

def load_encoder(model_name: str):
    model_dir = __get_model_dir(model_name)
    keras_model_path = os.path.join(model_dir, 'autoencoder.keras')
    if os.path.exists(keras_model_path):
        # 1) load the full AutoEncoder
        auto = keras.models.load_model(keras_model_path, compile=False)
        # 2) try to find the latent layer by name
        try:
            latent = auto.get_layer("ae_latent").output
        except ValueError:
            __logger.warning(
                f"Encoder not found. "
                f"Make sure your build() named it 'ae_latent'."
            )
            # fallback for legacy models where the encoder was layer[1]
            legacy_encoder = auto.layers[1]
            # assume layer[1] is a Model or Sequential
            return legacy_encoder

        # 3) new 2D-style model: build an encoder from auto.input → latent
        encoder2d = keras.Model(inputs=auto.input,
                        outputs=latent,
                        name=f"{model_name}_encoder2d")
        return encoder2d
    else:
        from keras.models import Sequential, model_from_json
        # from tensorflow.keras.models import model_from_json
        with open(os.path.join(model_dir, 'encoder.json'), 'r') as json_file:
            loaded_encoder = model_from_json(json_file.read(), custom_objects={"Sequential": Sequential})
            loaded_encoder.load_weights(os.path.join(model_dir, 'encoder.h5'))
        return loaded_encoder


def load_decoder(model_name: str):
    model_dir = __get_model_dir(model_name)
    keras_model_path = os.path.join(model_dir, 'autoencoder.keras')
    if os.path.exists(keras_model_path):
        # 1) load the combined autoencoder (no compile)
        auto    = keras.models.load_model(keras_model_path, compile=False)

        # 2) grab the nested decoder Model by name
        try:
            backbone = model_name.split("_")[0]
            decoder_layer = auto.get_layer(f"{backbone}_decoder")
        except ValueError:
            raise ValueError(
                f"Decoder not found. "
                f"Make sure your build() named it '{backbone}_decoder'."
            )

        # 3) If that layer is *already* a Model (keras.Model subclass), return it
        if isinstance(decoder_layer, keras.Model):
            return decoder_layer

        # 4) Otherwise wrap its input→output into a Model
        inp  = decoder_layer.input   # should be the latent-vector Input
        out  = decoder_layer.output  # should be the (H,W) output
        return keras.Model(inp, out, name=f"{model_name}_decoder2d_extracted")
    else:
        from keras.models import Sequential, model_from_json
        with open(os.path.join(model_dir, 'decoder.json'), 'r') as json_file:
            loaded_decoder = model_from_json(json_file.read(), custom_objects={"Sequential": Sequential})
            loaded_decoder.load_weights(os.path.join(model_dir, 'decoder.h5'))
        return loaded_decoder


def __get_model_dir(model_name):
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    return os.path.join(project_dir, "data", "models", model_name)
