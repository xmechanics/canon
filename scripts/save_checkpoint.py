import os
import sys
import keras

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import canon.autoencode

if __name__ == "__main__":
    model_name = canon.autoencode.AE_128_to_256
    checkpoint = "checkpoints/%s/autoencoder.1685.hdf5" % model_name

    autoencoder = keras.models.load_model(checkpoint)
    encoder = autoencoder.layers[1]
    decoder = autoencoder.layers[2]

    model_dir = "models/{}".format(model_name)

    # save trained weights
    with open(os.path.join(model_dir, "encoder.json"), "w") as json_file:
        json_file.write(encoder.to_json())
    with open(os.path.join(model_dir, "decoder.json"), "w") as json_file:
        json_file.write(decoder.to_json())
    encoder.save_weights(os.path.join(model_dir, "encoder.h5"))
    decoder.save_weights(os.path.join(model_dir, "decoder.h5"))
