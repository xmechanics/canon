import os
import yaml
import logging
import logging.config


def init_logging(cfg_path=None, level=logging.DEBUG):
    if cfg_path is not None and os.path.exists(cfg_path):
        with open(cfg_path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logFormatter = logging.Formatter("%(asctime)s [%(threadName)s] %(levelname)s %(name)s:%(lineno)d - %(message)s")
        rootLogger = logging.getLogger()
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)
        rootLogger.setLevel(level)


def init_mpi_logging(cfg_path=None, level=logging.DEBUG):
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
        if cfg_path is not None and os.path.exists(cfg_path):
            with open(cfg_path, 'rt') as f:
                content = f.read()
                content = content.replace("{{MPI_RANK}}", "[Process-" + str(MPI_RANK) + "]")
                config = yaml.safe_load(content)
            logging.config.dictConfig(config)
        else:
            if MPI_COMM.size == 1:
                logFormatter = logging.Formatter("%(asctime)s [%(threadName)s] %(levelname)s %(name)s:%(lineno)d - %(message)s")
            else:
                logFormatter = logging.Formatter("%(asctime)s [Process-" + str(MPI_RANK) + "] [%(threadName)s] %(levelname)s %(name)s:%(lineno)d - %(message)s")
            rootLogger = logging.getLogger()
            consoleHandler = logging.StreamHandler()
            consoleHandler.setFormatter(logFormatter)
            rootLogger.addHandler(consoleHandler)
            rootLogger.setLevel(level)
