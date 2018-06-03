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

from canon.pattern.latent_extractor import LatentExtractor
from canon.dat.datreader import read_dats, idx2XY, blacklist
from canon.pattern.model import GMModel, KMeansModel
from canon.pattern.labeler import SeqLabeler

from plotseq import plot_seq


read_file = read_dats
MODEL_NAME = "AE_20180602"
EXTRACTOR = LatentExtractor(MODEL_NAME)


def split_workload(loads, nproc):
    load_groups = [[] for _ in range(nproc)]
    for i, l in enumerate(loads):
        load_groups[i % len(load_groups)].append(l)
    return load_groups


def read_sample_tiffs(dir_path, NX, steps):
    if MPI_RANK == 0:
        t0 = timer()
        filenames = [filename for filename in os.listdir(dir_path)]
        _logger.info('Found %d files in the directory %s.' % (len(filenames), dir_path))
        sample_files = []
        for filename in filenames:
            if filename in blacklist:
                continue
            X, Y = idx2XY(int(filename[-9:-4]), NX)
            if X % steps[0] == 0 and Y % steps[1] == 0:
                sample_files.append(os.path.join(dir_path, filename))
        _logger.info('Selected %d sample files according to step size %s.' % (len(sample_files), steps))
        file_groups = split_workload(sample_files, MPI_COMM.size)
    else:
        file_groups = None

    file_names = MPI_COMM.scatter(file_groups, root=0)
    _logger.info('Assigned %d tiffs to read.' % len(file_names))

    t0_loc = timer()
    codes = EXTRACTOR.features(file_names)
    _logger.info('Got %d [local] tiffs. %g sec' % (len(codes), timer() - t0_loc))

    codes = MPI_COMM.gather(codes, root=0)
    if MPI_RANK == 0:
        codes = [t for g in codes for t in g]
        _logger.info('Gathered %d latent codes in total. %g sec' % (len(codes), timer() - t0))
        return codes


def split_patterns_mpi(patterns):
    if MPI_RANK == 0:
        pattern_groups = split_workload(patterns, MPI_COMM.size)
    else:
        pattern_groups = None
    group = MPI_COMM.scatter(pattern_groups, root=0)
    return group


def merge_fts_mpi(data_loc):
    data_stacked = MPI_COMM.gather(data_loc, root=0)
    if MPI_RANK == 0:
        data_merged = [o for l in data_stacked for o in l]
        logging.info('Gathered %d data points of %d features. Total size = %.2f MB' %
                      (len(data_merged), len(data_loc[0]), sys.getsizeof(data_merged) / (1024. ** 2)))
        return data_merged


def extract_features(extractor, sample_patterns):
    extractor = MPI_COMM.bcast(extractor, root=0)
    patterns_loc = split_patterns_mpi(sample_patterns)
    _logger.info('Assigned %d [local] patterns to extract features' % len(patterns_loc))

    t0_loc = timer()
    data_loc = map(extractor.features, patterns_loc)
    _logger.info('Extracted %d features x %d [local] patterns. %g sec' %
                  (len(data_loc[0]), len(patterns_loc), timer() - t0_loc))
    data = np.array(merge_fts_mpi(data_loc))
    if MPI_RANK == 0:
        _logger.info('Extracted %d features x %d patterns. %g sec' %
                     (len(data[0]), len(sample_patterns), timer() - t0_loc))

    return data


def score_dir(extractor, model, dir_path, limit=None, batch_size=100):
    if MPI_RANK == 0:
        filenames = [os.path.join(dir_path, filename) for filename in os.listdir(dir_path) if filename not in blacklist]
        _logger.info('Found %d files in the directory %s.' % (len(filenames), dir_path))
        limit = len(filenames) if limit is None else min(limit, len(filenames))
        file_groups = split_workload(filenames[:limit], MPI_COMM.size)
    else:
        file_groups = None
    filenames = MPI_COMM.scatter(file_groups, root=0)
    _logger.info('Received %d files to score.' % len(filenames))

    t0 = timer()
    scoresinds_loc = score_files_loc(extractor, model, filenames, batch_size=batch_size)
    scoresinds_stack = MPI_COMM.gather(scoresinds_loc, root=0)
    if MPI_RANK == 0:
        scoresinds = sum(scoresinds_stack, [])
        _logger.info('Scored %d patterns. %g sec' % (len(scoresinds), timer() - t0))
        return scoresinds


def score_files_loc(extractor, model, filenames, batch_size=None):
    t0 = timer()
    if batch_size is None:
        nbatches = 1
    else:
        nbatches = max(1, int(len(filenames) / batch_size))
    file_batches = split_workload(filenames, nbatches)
    _logger.info('Split files to be scored into %d batches.' % nbatches)

    scoreinds = []
    for file_batch in file_batches:
        scoreinds += score_batch(extractor, model, file_batch)
    _logger.info('Scored %d [local] patterns. %g sec' % (len(scoreinds), timer() - t0))

    return scoreinds


def score_batch(extractor, model, filenames):
    t0 = timer()
    indices = [int(f[-9:-4]) for f in filenames]
    scores = model.score(extractor.features(filenames))
    _logger.info('Scored a batch of %d [local] patterns, %d are [None]. %g sec'
                  % (len(filenames), sum(1 for s in scores if s is None), timer() - t0))
    return [(s, i) for (s, i) in zip(scores, indices) if s is not None]


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

    # CuAlNi_mart2
    # scratch = "/Users/sherrychen/scratch/"
    scratch = "."
    z_file = "Z.txt"
    z_plot = "Z"
    tiff_dir = os.path.join(scratch, "img", "CuAlNi_mart2_processed")
    seq_files = [os.path.join(scratch, "seq", "CuAlNi_mart2_.SEQ")]

    NX = 40
    NY = 100
    step = (5, 5)
    training_set = read_sample_tiffs(tiff_dir, NX, (2, 2))    # sample_patterns on lives on core-0

    if MPI_RANK == 0:
        model = GMModel()
        model.train(np.array(training_set), n_clusters=6, preprocessors=[StandardScaler(), PCA(whiten=True)])
    else:
        model = None
    model = MPI_COMM.bcast(model, root=0)

    score_inds = score_dir(EXTRACTOR, model, tiff_dir, limit=None, batch_size=200)

    if MPI_RANK == 0:
        labeler = SeqLabeler(seq_files)
    else:
        labeler = None
    labeler = MPI_COMM.bcast(labeler, root=0)
    score_inds = relabel(labeler, score_inds)

    if MPI_RANK == 0:
        Z = np.empty([NY, NX])
        Z[:] = np.nan
        for score, idx in score_inds:
            if score is not None:
                ix, iy = idx2XY(idx, NX)
                if ix < NY:
                    Z[ix, iy] = score
        _logger.info('Z matrix has %d nans' % sum(1 for row in Z for z in row if np.isnan(z)))
        np.savetxt(z_file, Z)
        _logger.info('Write Z matrix into ' + z_file + ' in ' + os.path.dirname(os.path.abspath(__file__)))

    # Visualization
    if MPI_RANK == 0:
        Z = np.loadtxt(z_file)
        plot_seq(Z, step, colormap='jet', filename=os.path.join(scratch, "img", z_plot))


