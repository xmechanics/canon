import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import canon
from canon.autoencode import reset_tf_session
from canon.autoencode.train import train


if __name__ == "__main__":
    nersc = ("IN_NERSC" in os.environ) and os.environ["IN_NERSC"] == "true"
    s = reset_tf_session(nersc=nersc)
    architecture = canon.autoencode.AE_128_to_256

    # train from scratch
    train(architecture, "img/processed_128", "img/test_128", run_number="conv_4_dense_2", verbose=1)

    # # train from checkpoint
    # train(architecture, "img/processed_128", "img/test_128",
    #       initial_epoch=1437,
    #       verbose=1,
    #       run_number="conv_4_dense_2_dropout",
    #       checkpoint="checkpoints/AE_128_to_256/conv_4_dense_2_dropout/autoencoder.1437.hdf5")
