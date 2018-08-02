import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import canon
from canon.autoencode import reset_tf_session
from canon.autoencode.train import train


if __name__ == "__main__":
    from canon.common.init import init_logging
    init_logging()

    nersc = ("IN_NERSC" in os.environ) and os.environ["IN_NERSC"] == "true"
    s = reset_tf_session(nersc=nersc)
    architecture = canon.autoencode.CONV_4
    n_features = 32
    train(architecture, n_features, "img/processed_981", "img/test_981", verbose=1, dryrun=False)
