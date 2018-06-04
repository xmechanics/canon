import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import canon
from canon.autoencode.train import train

if __name__ == "__main__":
    model_name = canon.autoencode.AE_128_to_256

    # train from scratch
    # train(model_name, "img/processed_128")

    # train from checkpoint
    train(model_name, "img/processed_128",
          initial_epoch=1000,
          checkpoint="checkpoints/AE_128_to_256/autoencoder.999.hdf5")
