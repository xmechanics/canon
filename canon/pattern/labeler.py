from __future__ import absolute_import
from canon.seq.seqreader import SeqReader


# class SeqLabeler:

    # def __init__(self, seqfiles):
    #     Z_merged, N_merged = None, None
    #     for seqfile in seqfiles:
    #         seq = read_seq(seqfile)
    #         Z, _, N = get_Zmap(seq, 'orsnr___')
    #         if Z_merged is None:
    #             Z_merged, N_merged = Z, N
    #         else:
    #             Z_merged, N_merged = merge_Zmap(Z, Z_merged, N, N_merged)
    #     self.__Z = Z_merged
    #
    # def evaluate(self, ix, iy):
    #     z = self.__Z[ix, iy]
    #     return z if z != 0 else None
