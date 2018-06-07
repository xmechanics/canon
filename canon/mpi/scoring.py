import os
import sys
import logging
import numpy as np
from skimage.io import imread
from timeit import default_timer as timer
from mpi4py import MPI

MPI_COMM = MPI.COMM_WORLD
MPI_RANK = MPI_COMM.Get_rank()

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
_logger = logging.getLogger(__name__)


from canon.pattern.feature_extractor import FeaturesExtractor
from canon.pattern.model import Model
from canon.util import split_workload


def score_dir(extractor: FeaturesExtractor, model: Model, dir_path, limit=None, batch_size=100, blacklist=[]):
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
    img_data = np.array([imread(f) for f in filenames])
    scores = model.score(extractor.features(img_data))
    _logger.info('Scored a batch of %d [local] patterns, %d are [None]. %g sec'
                  % (len(filenames), sum(1 for s in scores if s is None), timer() - t0))

    return [(s, i) for (s, i) in zip(scores, indices) if s is not None]

