import pytest
from canon.seq.seqreader import SeqReader
from .. import resource


def test_read_seq():
    reader = SeqReader(resource('seq/Quartz_500Mpa_.SEQ'))
    reader.get_Om()
    Z, _, N = reader.get_Zmap('orsnr___')


def test_merge_Zmap():
    reader = SeqReader()

    reader.read_seq(resource('seq/au30_a1_.SEQ'))
    Z1, _, N1 = reader.get_Zmap('orsnr___')

    reader.read_seq(resource('seq/au30_m1_.SEQ'))
    Z2, _, N2 = reader.get_Zmap('orsnr___')

    Z, N = SeqReader.merge_Zmap(Z1, Z2, N1, N2)


if __name__ == '__main__':
    pytest.main()
