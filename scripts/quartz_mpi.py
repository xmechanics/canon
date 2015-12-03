from canon.mpi.init import *

import numpy as np
import os
import sys
from timeit import default_timer as timer
from itertools import groupby

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from canon.dat.datreader import read_dats, read_txt, idx2XY, blacklist
from canon.pattern.feature_extractor import AllPeaksExtractor, PeaksNumberExtractor, CombinedExtractor
from canon.pattern.model import GMMModel
from canon.pattern.labeler import SeqLabeler


read_file = read_dats


def split_workload(loads, nproc):
    load_groups = [[] for _ in xrange(nproc)]
    for i, l in enumerate(loads):
        load_groups[i % len(load_groups)].append(l)
    return load_groups


def read_sample_patterns(dir_path, NX, step):
    if MPI_RANK == 0:
        t0 = timer()
        filenames = [filename for filename in os.listdir(dir_path)]
        logging.debug('Found %d files in the directory %s.' % (len(filenames), dir_path))
        sample_files = []
        for filename in filenames:
            if filename in blacklist:
                continue
            X, Y = idx2XY(int(filename[-9:-4]), NX)
            if X % step[0] == 0 and Y % step[1] == 0:
                sample_files.append(os.path.join(dir_path, filename))
        logging.debug('Selected %d sample files according to step size %s.' % (len(sample_files), step))
        file_groups = split_workload(sample_files, MPI_COMM.size)
    else:
        file_groups = None

    filenames = MPI_COMM.scatter(file_groups, root=0)
    logging.debug('Assigned %d DAT files to read.' % len(filenames))

    t0_loc = timer()
    patterns = read_file(filenames)
    logging.debug('Got %d [local] sample patterns. %g sec' % (len(patterns), timer() - t0_loc))

    patterns = MPI_COMM.gather(patterns, root=0)
    if MPI_RANK == 0:
        patterns = [t for g in patterns for t in g]
        logging.info('Gathered %d sample patterns in total. %g sec' % (len(patterns), timer() - t0))
        return patterns


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
    logging.debug('Assigned %d [local] patterns to extract features' % len(patterns_loc))

    t0_loc = timer()
    data_loc = map(extractor.features, patterns_loc)
    logging.debug('Extracted %d features x %d [local] patterns. %g sec' %
                  (len(data_loc[0]), len(patterns_loc), timer() - t0_loc))
    data = np.array(merge_fts_mpi(data_loc))
    if MPI_RANK == 0:
        logging.info('Extracted %d features x %d patterns. %g sec' %
                     (len(data[0]), len(sample_patterns), timer() - t0_loc))

    return data


def score_dir(extractor, model, dir_path, limit=None, batch_size=100):
    if MPI_RANK == 0:
        filenames = [os.path.join(dir_path, filename) for filename in os.listdir(dir_path) if filename not in blacklist]
        logging.debug('Found %d files in the directory %s.' % (len(filenames), dir_path))
        limit = len(filenames) if limit is None else min(limit, len(filenames))
        file_groups = split_workload(filenames[:limit], MPI_COMM.size)
    else:
        file_groups = None
    filenames = MPI_COMM.scatter(file_groups, root=0)
    logging.debug('Received %d files to score.' % len(filenames))

    t0 = timer()
    scoresinds_loc = score_files_loc(extractor, model, filenames, batch_size=batch_size)
    scoresinds_stack = MPI_COMM.gather(scoresinds_loc, root=0)
    if MPI_RANK == 0:
        scoresinds = sum(scoresinds_stack, [])
        logging.info('Scored %d patterns. %g sec' % (len(scoresinds), timer() - t0))
        return scoresinds


def score_files_loc(extractor, model, filenames, batch_size=None):
    t0 = timer()
    if batch_size is None:
        nbatches = 1
    else:
        nbatches = max(1, int(len(filenames) / batch_size))
    file_batches = split_workload(filenames, nbatches)
    logging.debug('Split files to be scored into %d batches.' % nbatches)

    scoreinds = []
    for file_batch in file_batches:
        scoreinds += score_batch(extractor, model, file_batch)
    logging.debug('Scored %d [local] patterns. %g sec' % (len(scoreinds), timer() - t0))

    return scoreinds


def score_batch(extractor, model, filenames):
    t0 = timer()
    indices = [int(f[-9:-4]) for f in filenames]
    patterns = read_file(filenames)
    scores = model.score(np.array(map(extractor.features, patterns)))
    logging.debug('Scored a batch of %d [local] patterns, %d are [None]. %g sec'
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
    logging.debug('Got %d [local] groups of scores to re-label, which is %d in total.' %
                  (len(sub_groups), sum(map(len, sub_groups))))

    new_scoreinds_loc = relabel_score_groups(labeler, sub_groups)
    new_scoreinds_stack = MPI_COMM.gather(new_scoreinds_loc, root=0)
    if MPI_RANK == 0:
        new_scoreinds = sum(new_scoreinds_stack, [])
        logging.info('Re-labeled %d scores. %g sec' % (len(new_scoreinds), timer() - t0))
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
            logging.warn('%d scores in cluster %d are re-labeled to [None]!' % (len(scoreinds), scoreinds[0][0][0]))
    logging.info('Re-labeled %d [local] scores. %g sec' % (sum(map(len, groups)), timer() - t0))
    return new_scoreinds


if __name__ == '__main__':
    # Au30
    read_file = read_txt
    case_name = 'au30_area2_2k'
    scratch = "/Users/sherrychen/scratch/"
    dir_path = scratch + "peaks/txt/" + case_name
    seq_files = [scratch + "seqfiles/" + f for f in ('au30_a1_.SEQ', 'au30_m1_.SEQ')] # 'm4fine_a.SEQ',
    all_peaks_threshold = 0.8
    # if MPI_RANK == 0:
    #     labeler = SeqLabeler(seq_files)
    NX = 100
    NY = 20
    step = (2, 2)
    sample_patterns = read_sample_patterns(dir_path, NX, (1, 1))    # sample_patterns on lives on core-0

    # # quartz
    # case_name = 'quartz_500mpa'
    # scratch = "/Users/sherrychen/scratch/"
    # dir_path = scratch + "peaks/dat/" + case_name
    # seq_files = (scratch + "seqfiles/" + 'Quartz_500Mpa_.SEQ', )
    # NX = 120
    # NY = 120
    # step = (5, 5)
    # all_peaks_threshold = 0.9
    # sample_patterns = read_sample_patterns(dir_path, NX, (4, 4))    # sample_patterns on lives on core-0

    if MPI_RANK == 0:
        t0 = timer()
        extractor1 = AllPeaksExtractor(sample_patterns, intensity_threshold=all_peaks_threshold, gaussion_height=1, gaussian_width=5)
        extractor2 = PeaksNumberExtractor(intensity_threshold=0.0)
        extractor = CombinedExtractor([extractor2, extractor1])
        # extractor = extractor2
        logging.info("Constructed a feature extractor. %g sec" % (timer() - t0))
    else:
        extractor = None
    data = extract_features(extractor, sample_patterns)
    extractor = MPI_COMM.bcast(extractor, root=0)

    if MPI_RANK == 0:
        model = GMMModel()
        model.train(np.array(data), preprocessors=[StandardScaler()])
        # model.train(np.array(data), preprocessors=[StandardScaler(), PCA(whiten=True)])
    else:
        model = None
    model = MPI_COMM.bcast(model, root=0)

    scoreinds = score_dir(extractor, model, dir_path, limit=None, batch_size=200)

    if MPI_RANK == 0:
        labeler = SeqLabeler(seq_files)
    else:
        labeler = None
    labeler = MPI_COMM.bcast(labeler, root=0)
    scoreinds = relabel(labeler, scoreinds)

    if MPI_RANK == 0:
        Z = np.empty([NY, NX])
        Z[:] = np.nan
        for score, idx in scoreinds:
            if score is not None:
                ix, iy = idx2XY(idx, NX)
                if ix < NY:
                    Z[ix, iy] = score
        logging.debug('Z matrix has %d nans' % sum(1 for row in Z for z in row if np.isnan(z)))
        np.savetxt('Z.txt', Z)
        logging.info('Write Z matrix into Z.txt in ' + os.path.dirname(os.path.abspath(__file__)))

        from plotseq import plot_seq
        # # Z = np.loadtxt('Z.txt')
        plot_seq(Z, step, colormap='jet', filename=scratch + "img/clustering_" + case_name)


