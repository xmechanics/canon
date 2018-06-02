import os
import sys
import logging
import numpy as np
from PIL import Image
from mpi4py import MPI
from timeit import default_timer as timer

MPI_COMM = MPI.COMM_WORLD
MPI_RANK = MPI_COMM.Get_rank()

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
_logger = logging.getLogger(__name__)

import canon
from canon.util import split_workload


def process_images(input_dir, output_dir):
    if MPI_RANK == 0:
        os.makedirs(output_dir, 777, exist_ok=True)
        t0 = timer()
        filenames = [filename for filename in os.listdir(input_dir)]
        num_images = len(filenames)
        _logger.info('Found %d files in the directory %s.' % (len(filenames), input_dir))
        file_groups = split_workload(filenames, MPI_COMM.size)
    else:
        file_groups = None

    filenames = MPI_COMM.scatter(file_groups, root=0)
    _logger.info('Assigned %d image files to process.' % len(filenames))
    t0_loc = timer()
    for i, tiff in enumerate(filenames):
        jpg = tiff.lower().replace(".tiff", ".jpg").replace(".tif", ".jpg")

        reader = canon.TiffReader(canon.TiffReader.PILATUS)
        reader.loadtiff(os.path.join(input_dir, tiff))
        reader.remove_background()
        reader.whitening()
        data = reader.image()

        img = Image.fromarray(data.astype(np.uint8))
        img.save(os.path.join(output_dir, jpg))

        pct = 100. * (i + 1.) / len(filenames)
        _logger.info('Processed %d / %d (%.2f%%) [local] images. %g sec' % (i + 1, len(filenames), pct, timer() - t0_loc))

    _logger.info('Processed %d [local] images in total. %g sec' % (len(filenames), timer() - t0_loc))

    if MPI_RANK == 0:
        # noinspection PyUnboundLocalVariable
        _logger.info('Processed %d images in total. %g sec' % (num_images, timer() - t0))


if __name__ == '__main__':
    from canon.common.init import init_mpi_logging
    init_mpi_logging("logging_mpi.yaml")

    process_images("img/test", "img/processed")

