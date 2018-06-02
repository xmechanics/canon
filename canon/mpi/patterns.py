import logging
import numpy as np
from timeit import default_timer as timer

from canon.mpi.init import *
from canon.util import split_workload, merge_workload
from canon.dat.datreader import read_dats

_logger = logging.getLogger(__name__)


def distribute_mpi(works_global):
    if MPI_RANK == 0:
        works_grps = split_workload(works_global, MPI_COMM.size)
    else:
        works_grps = None
    works_local = MPI_COMM.scatter(works_grps, root=0)
    return works_local


def gather_mpi(works_local):
    works = MPI_COMM.gather(works_local, root=0)
    if MPI_RANK == 0:
        works_global = merge_workload(works)
        return works_global


def read_patterns(filenames, read_file_func=read_dats):
    t0 = timer()
    if MPI_RANK == 0:
        _logger.debug('Got %d files to read patterns.' % len(filenames))
    files_in_group = distribute_mpi(filenames)
    _logger.debug('Assigned %d [local] files to read patterns.' % len(files_in_group))

    t0_loc = timer()
    patterns = read_file_func(files_in_group)
    _logger.debug('Got %d [local] sample patterns. %g sec' % (len(patterns), timer() - t0_loc))

    patterns = gather_mpi(patterns)
    if MPI_RANK == 0:
        _logger.info('Gathered %d sample patterns in total. %g sec' % (len(patterns), timer() - t0))
        return patterns


def extract_features(extractor, sample_patterns):
    extractor = MPI_COMM.bcast(extractor, root=0)
    patterns_loc = distribute_mpi(sample_patterns)
    _logger.debug('Assigned %d [local] patterns to extract features' % len(patterns_loc))

    t0_loc = timer()
    data_loc = map(extractor.features, patterns_loc)
    _logger.debug('Extracted %d features x %d [local] patterns. %g sec' %
                  (len(data_loc[0]), len(patterns_loc), timer() - t0_loc))
    data = np.array(gather_mpi(data_loc))
    if MPI_RANK == 0:
        _logger.info('Extracted %d features x %d patterns. %g sec' %
                     (len(data[0]), len(sample_patterns), timer() - t0_loc))
    return data

