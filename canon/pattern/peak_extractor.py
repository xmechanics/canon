import logging
import numpy as np
from canon.pattern.feature_extractor import FeaturesExtractor
from canon.img.peaks import num_peaks

_logger = logging.getLogger(__name__)


class PeakNumberExtractor(FeaturesExtractor):

    def features(self, img_data, skip_normalize=True):
        return np.array([[num_peaks(img)] for img in img_data]).astype("float32")


if __name__ == "__main__":
    from canon.common.init import init_mpi_logging
    from skimage.io import imread
    init_mpi_logging("logging.yaml")
    extractor = PeakNumberExtractor()
    images = [
        "../../scripts/img/CuAlNi_mart2_processed/cualni_mart2_00001.jpg",
        "../../scripts/img/CuAlNi_mart2_processed/cualni_mart2_00002.jpg"
    ]
    img_data = [imread(f) for f in images]
    print(extractor.features(img_data))
