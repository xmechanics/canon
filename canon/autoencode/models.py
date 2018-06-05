import os
from keras.models import model_from_json

AE_128_to_256 = "AE_128_to_256"
AE_128_to_64 = "AE_128_to_64"


def load_encoder(model_name: str):
    model_dir = __get_model_dir(model_name)
    with open(os.path.join(model_dir, 'encoder.json'), 'r') as json_file:
        loaded_encoder = model_from_json(json_file.read())
        loaded_encoder.load_weights(os.path.join(model_dir, 'encoder.h5'))
    return loaded_encoder


def load_decoder(model_name: str):
    model_dir = __get_model_dir(model_name)
    with open(os.path.join(model_dir, 'decoder.json'), 'r') as json_file:
        loaded_decoder = model_from_json(json_file.read())
        loaded_decoder.load_weights(os.path.join(model_dir, 'decoder.h5'))
    return loaded_decoder


def __get_model_dir(model_name):
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    return os.path.join(project_dir, "data", "models", model_name)
