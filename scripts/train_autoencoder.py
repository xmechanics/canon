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
    architecture = canon.autoencode.AE_128_to_256
    run_number = "conv_4"
    train(architecture, run_number, "img/processed_981", "img/test_981", verbose=1, dryrun=False)
