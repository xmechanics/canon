import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import canon
from canon.autoencode import reset_tf_session
from canon.autoencode.train import train

if __name__ == "__main__":
	s = reset_tf_session(nersc=False)

    model_name = canon.autoencode.AE_128_to_256

    # train from scratch
    # train(model_name, "img/processed_128")

    # train from checkpoint
    train(model_name, "img/processed_128",
         initial_epoch=1686,
         checkpoint="checkpoints/AE_128_to_256/autoencoder.1685.hdf5")
