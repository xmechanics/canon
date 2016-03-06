from __future__ import absolute_import
from canon.seq.seqreader import SeqReader


class SeqLabeler:

    def __init__(self, seqfiles):
        Z_merged, N_merged = None, None
        readers = map(SeqReader, seqfiles)
        for reader in readers:
            Z, _, N = reader.get_Zmap('orsnr___', thres=10)
            if Z_merged is None:
                Z_merged, N_merged = Z, N
            else:
                Z_merged, N_merged = SeqReader.merge_Zmap(Z, Z_merged, N, N_merged)
        self.__Z = Z_merged
        self.__NX = len(self.__Z[0])
        self.__NY = len(self.__Z)

    def evaluate(self, idx):
        iy, ix = self.idx2XY(idx)
        if ix >= self.__NX or iy >= self.__NY:
            return None
        z = self.__Z[iy, ix]
        return z if z != 0.0 else None

    def idx2XY(self, idx):
        return int((idx - 1) / self.__NX), (idx - 1) % self.__NX
