import os
import sys
import logging
from timeit import default_timer as timer
from itertools import groupby
from mpi4py import MPI
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

MPI_COMM = MPI.COMM_WORLD
MPI_RANK = MPI_COMM.Get_rank()

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
_logger = logging.getLogger(__name__)

from canon.pattern import LatentExtractor, PeakNumberExtractor, CombinedExtractor
from canon.dat.datreader import read_dats, idx2XY
from canon.pattern.model import GMModel, KMeansModel, BGMModel, MeanShiftModel
from canon.pattern.labeler import SeqLabeler
from canon.util import split_workload
from canon.mpi import extract_sample_features, score_dir

from plotseq import plot_seq


read_file = read_dats


def relabel(labeler, scoreinds):
    t0 = timer()
    # split scores into groups by label
    if MPI_RANK == 0:
        groups = []
        for k, g in groupby(sorted(scoreinds, key=lambda si: si[0][0]), key=lambda si: si[0][0]):
            groups.append(list(g))
        sub_groups = split_workload(sorted(groups, key=len), MPI_COMM.size)
    else:
        sub_groups = 0
    sub_groups = MPI_COMM.scatter(sub_groups, root=0)
    _logger.info('Got %d [local] groups of scores to re-label, which is %d in total.' %
                  (len(sub_groups), sum(map(len, sub_groups))))

    new_scoreinds_loc = relabel_score_groups(labeler, sub_groups)
    new_scoreinds_stack = MPI_COMM.gather(new_scoreinds_loc, root=0)
    if MPI_RANK == 0:
        new_scoreinds = sum(new_scoreinds_stack, [])
        _logger.info('Re-labeled %d scores. %g sec' % (len(new_scoreinds), timer() - t0))
        return new_scoreinds


def relabel_score_groups(labeler, groups):
    t0 = timer()
    new_scoreinds = []
    for scoreinds in groups:
        scoreinds = [si for si in scoreinds if si[0] is not None]
        scoreinds = sorted(scoreinds, key=lambda si: si[0][1], reverse=True)
        weighted_scores = []
        for si in scoreinds[:min(len(scoreinds), 1000)]:
            score = labeler.evaluate(si[1])
            if score is not None:
                weighted_scores.append((score, si[0][1]))
        centroid_score = np.sum([s * w for s, w in weighted_scores])/np.sum([w for _, w in weighted_scores]) \
            if len(weighted_scores) > 0 else None
        new_scoreinds += [(centroid_score, si[1]) for si in scoreinds]
        if centroid_score is None:
            _logger.warning('%d scores in cluster %d are re-labeled to [None]!' % (len(scoreinds), scoreinds[0][0][0]))
    _logger.info('Re-labeled %d [local] scores. %g sec' % (sum(map(len, groups)), timer() - t0))
    return new_scoreinds


if __name__ == '__main__':
    from canon.common.init import init_mpi_logging
    init_mpi_logging("logging_mpi.yaml")

    extractor1 = PeakNumberExtractor()
    extractor2 = LatentExtractor("AE_128_256_20180606")
    extractor = CombinedExtractor([extractor1, extractor2])
    extractor = extractor2

    # CuAlNi_mart2
    # scratch = "/Users/sherrychen/scratch/"
    scratch = "."
    z_file = "Z.txt"
    z_plot = "Z"
    # tiff_dir = os.path.join(scratch, "img", "C_2_1_test_processed")
    # seq_files = [os.path.join(scratch, "seq", "C_2_1_test_.SEQ")]
    # NX = 25
    # NY = 20
    # sample_rate = 1.0

    tiff_dir = os.path.join(scratch, "img", "CuAlNi_mart2_processed")
    seq_files = [os.path.join(scratch, "seq", "CuAlNi_mart2_.SEQ")]
    NX = 100
    NY = 80
    sample_rate = 1.0

    # tiff_dir = os.path.join(scratch, "img", "ZrO2_770C_wb1_processed")
    # seq_files = [os.path.join(scratch, "seq", "CuAlNi_mart2_.SEQ")]
    # NX = 110
    # NY = 80
    # sample_rate = 1.0

    step = (5, 5)
    training_set = extract_sample_features(extractor, tiff_dir, sample_rate=sample_rate)    # sample_patterns on lives on core-0

    if MPI_RANK == 0:
        model = BGMModel()
        model.train(np.array(training_set), preprocessors=[StandardScaler(), PCA(whiten=True)])
        # model = DBSCANModel(eps=10, min_samples=10)
        # model.train(np.array(training_set), preprocessors=[StandardScaler()])
    else:
        model = None
    model = MPI_COMM.bcast(model, root=0)

    score_inds = score_dir(extractor, model, tiff_dir, limit=None, batch_size=200)

    # # only support relabeling Guassian Mixture based model for now
    # if MPI_RANK == 0:
    #     labeler = SeqLabeler(seq_files)
    # else:
    #     labeler = None
    # labeler = MPI_COMM.bcast(labeler, root=0)
    # score_inds = relabel(labeler, score_inds)

    if MPI_RANK == 0:
        Z = np.empty([NY, NX])
        Z[:] = np.nan
        for score, idx in score_inds:
            if score is not None:
                ix, iy = idx2XY(idx, NX)
                if ix < NY:
                    if isinstance(score, tuple) or isinstance(score, list) or isinstance(score, np.ndarray):
                        Z[ix, iy] = score[0]
                    else:
                        Z[ix, iy] = score
        _logger.info('Z matrix has %d nans' % sum(1 for row in Z for z in row if np.isnan(z)))
        np.savetxt(z_file, Z)
        _logger.info('Write Z matrix into ' + z_file + ' in ' + os.path.dirname(os.path.abspath(__file__)))

    # Visualization
    if MPI_RANK == 0:
        Z = np.loadtxt(z_file)
        scaler = model.get_label_scaler()
        plot_seq(scaler(Z), step, colormap='jet', filename=os.path.join(scratch, "img", z_plot))


