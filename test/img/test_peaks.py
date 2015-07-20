import pytest
import canon

from test import resource
from canon.img.peaks import find_peaks


@pytest.fixture
def pillar5():
    reader = canon.TiffReader('pilatus')
    reader.loadtiff(resource('Pillar5_00001.tif'))
    reader.fill_black()
    reader.remove_background()
    return reader.image()


def test_all_peaks(pillar5):
    peaks = find_peaks(pillar5)
    assert len(peaks) > 15


@pytest.fixture(params=[15, 10, 5])
def peaks_provider(request):
    return find_peaks(pillar5(), request.param), request.param


def test_peaks(peaks_provider):
    peaks, npeaks = peaks_provider
    assert abs(len(peaks) - npeaks) < 1

if __name__ == '__main__':
    pytest.main()