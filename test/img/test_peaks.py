import pytest
import canon

from .. import resource
from canon.img.peaks import find_peaks


@pytest.fixture
def pillar5():
    reader = canon.TiffReader('pilatus')
    reader.loadtiff(resource('peak_martensite.tif'))
    reader.remove_background()
    return reader.image()


def test_all_peaks(pillar5):
    peaks = find_peaks(pillar5)
    assert len(peaks) > 50


@pytest.fixture(params=[50, 40, 30, 20, 10])
def peaks_provider(request):
    return find_peaks(pillar5(), request.param), request.param


def test_peaks(peaks_provider):
    peaks, npeaks = peaks_provider
    assert abs(len(peaks) - npeaks) < 2

if __name__ == '__main__':
    pytest.main()