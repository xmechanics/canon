import pytest
from canon import TiffReader


@pytest.fixture(scope="module")
def pilatus():
    return TiffReader('pilatus')
