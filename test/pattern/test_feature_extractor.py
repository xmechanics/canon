import pytest
import numpy as np
from canon.pattern.feature_extractor import AllPeaksExtractor, PeaksNumberExtractor

pattern = np.array([
    [100, 100, 100],
    [100, 200, 90],
    [200, 100, 80],
    [200, 200, 70]
])

pattern2 = np.array([
    [101, 101, 100],
    [103, 203, 90],
    [210, 120, 80],
    [240, 170, 70]
])


def test_all_peaks():
    extractor = AllPeaksExtractor([pattern, pattern2])
    features = extractor.features(pattern)
    assert len(features) > 3


def test_peak_num():
    extractor = PeaksNumberExtractor(intensity_threshold=0.75)
    features = extractor.features(pattern)
    assert len(features) == 1


if __name__ == '__main__':
    pytest.main()
