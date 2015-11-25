# ==================================================
# BEGIN - MPI settings
# ==================================================
import logging
from mpi4py import MPI

try:
    # noinspection PyUnresolvedReferences
    MPI_INITIALIZED
except NameError:
    MPI_INITIALIZED = False

if not MPI_INITIALIZED:
    MPI_COMM = MPI.COMM_WORLD
    MPI_RANK = MPI_COMM.Get_rank()

    # log to console
    if MPI_COMM.size == 1:
        logFormatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    else:
        logFormatter = logging.Formatter("%(asctime)s [%(levelname)s] [Process-" + str(MPI_RANK) + "] %(message)s")
    rootLogger = logging.getLogger()
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.DEBUG)
# ==================================================
# END - MPI settings
# ==================================================
