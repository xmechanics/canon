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
from canon.util import split_workload, idx2XY


def extract_sample_features(extractor: FeaturesExtractor, dir_path, NX, steps, blacklist=[]):
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
    img_data = np.array([imread(f) for f in file_names])
    codes = extractor.features(img_data)
    _logger.info('Got %d [local] tiffs. %g sec' % (len(codes), timer() - t0_loc))

    codes = MPI_COMM.gather(codes, root=0)
    if MPI_RANK == 0:
        codes = [t for g in codes for t in g]
        _logger.info('Gathered %d latent codes in total. %g sec' % (len(codes), timer() - t0))
        return np.array(codes)