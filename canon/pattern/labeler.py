from __future__ import absolute_import
from canon.seq.seqreader import SeqReader


class SeqLabeler:

    def __init__(self, seqfiles):
        Z_merged, N_merged = None, None
        readers = map(SeqReader, seqfiles)
        for reader in readers:
            Z, N = reader.get_Zmap('orsnr___')
            if Z_merged is None:
                Z_merged, N_merged = Z, N
            else:
                Z_merged, N_merged = SeqReader.merge_Zmap(Z, Z_merged, N, N_merged)
        self.__Z = Z_merged
        self.__NX = len(self.__Z[0])
        self.__NY = len(self.__Z)

    def Z_map(self):
        return self.__Z

    def img_shape(self):
        return self.__NY, self.__NX

    def evaluate(self, idx):
        iy, ix = self.__idx2XY(idx)
        if ix >= self.__NX or iy >= self.__NY:
            return None
        z = self.__Z[iy, ix]
        return z if z != 0.0 else None

    def __idx2XY(self, idx):
        return int((idx - 1) / self.__NX), (idx - 1) % self.__NX
