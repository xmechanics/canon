import os
import sys
import logging
import numpy as np
from PIL import Image
from skimage.transform import resize
from mpi4py import MPI
from timeit import default_timer as timer

MPI_COMM = MPI.COMM_WORLD
MPI_RANK = MPI_COMM.Get_rank()

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
_logger = logging.getLogger(__name__)

import canon
from canon.util import split_workload


def get_file_names(img_dir, samples=-1):
    file_names = [os.path.join(img_dir, filename) for filename in os.listdir(img_dir)]
    if samples > 0:
        return np.random.choice(file_names, samples).tolist()
    else:
        return file_names


def process_images(file_paths, output_dir):
    if MPI_RANK == 0:
        os.makedirs(output_dir, exist_ok=True)
        t0 = timer()
        num_images = len(file_paths)
        _logger.info('Going to process %d files.' % len(file_paths))
        file_groups = split_workload(file_paths, MPI_COMM.size)
    else:
        file_groups = None

    filenames = MPI_COMM.scatter(file_groups, root=0)
    _logger.info('Assigned %d image files to process.' % len(filenames))
    t0_loc = timer()
    for i, tiff in enumerate(filenames):
        jpg = tiff.lower().split("/")[-1].replace(".tiff", ".jpg").replace(".tif", ".jpg")

        reader = canon.TiffReader(canon.TiffReader.PILATUS)
        reader.loadtiff(tiff)
        reader.remove_background()
        reader.normalize()
        data = reader.image()
        data = data[:981, :]
        data = resize(data, (128, 128), mode='reflect')
        img = Image.fromarray(data.astype(np.uint8))
        img.save(os.path.join(output_dir, jpg))

        if i % 10 == 0:
            pct = 100. * (i + 1.) / len(filenames)
            _logger.info(
                'Processed %d / %d (%.2f%%) [local] images. %g sec' % (i + 1, len(filenames), pct, timer() - t0_loc))

    _logger.info('Processed %d [local] images in total. %g sec' % (len(filenames), timer() - t0_loc))

    if MPI_RANK == 0:
        # noinspection PyUnboundLocalVariable
        _logger.info('Processed %d images in total. %g sec' % (num_images, timer() - t0))


if __name__ == '__main__':
    from canon.common.init import init_mpi_logging

    init_mpi_logging("logging_mpi.yaml")

    file_names = []
    if MPI_RANK == 0:
        for d in os.listdir("/Volumes/G-DRIVE/xmax_tiff"):
            if d != "CuAlMn_Dec2017" and d[0] != '.' and d[0] != "$" and d[-4:] != ".cal":
                file_names += get_file_names("/Volumes/G-DRIVE/xmax_tiff" + d, samples=200)
        for d in os.listdir("/Volumes/G-DRIVE/xmax_tiff/CuAlMn_Dec2017"):
            file_names += get_file_names("/Volumes/G-DRIVE/xmax_tiff/CuAlMn_Dec2017" + d, samples=200)
        _logger.info("Collected %d files" % len(file_names))
    # if MPI_RANK == 0:
    #     file_names = get_file_names("/Volumes/G-DRIVE/xmax_tiff/CuAlMn_Dec2017/C_2_1_test")
    process_images(file_names, "img/processed")
