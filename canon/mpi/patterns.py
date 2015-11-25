from timeit import default_timer as timer
from canon.mpi.init import *
from canon.util import split_workload
from canon.dat.datreader import read_dats


def read_patterns(filenames, read_file_func=read_dats):
    t0 = timer()
    if MPI_RANK == 0:
        logging.debug('Got %d files to read patterns.' % len(filenames))
        file_groups = split_workload(filenames, MPI_COMM.size)
    else:
        file_groups = None

    files_in_group = MPI_COMM.scatter(file_groups, root=0)
    logging.debug('Assigned %d [local] files to read patterns.' % len(files_in_group))

    t0_loc = timer()
    patterns = read_file_func(files_in_group)
    logging.debug('Got %d [local] sample patterns. %g sec' % (len(patterns), timer() - t0_loc))

    patterns = MPI_COMM.gather(patterns, root=0)
    if MPI_RANK == 0:
        patterns = [t for g in patterns for t in g]
        logging.info('Gathered %d sample patterns in total. %g sec' % (len(patterns), timer() - t0))
        return patterns
