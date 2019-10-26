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


def to_jpg_name(tiff_name):
    return tiff_name.lower().split("/")[-1].replace(".tiff", ".jpg").replace(".tif", ".jpg")


def get_existing_names(dirs):
    existing_names = set()
    for target_dir in dirs:
        existing_names |= set([filename for filename in os.listdir(target_dir)])
        _logger.info("Found {} files in target dirs".format(len(existing_names)))
    return existing_names


def get_file_names(img_dir, sample_rate=0.1, existing_names=set()):
    file_names = []
    files_in_subdir = []
    for filename in os.listdir(img_dir):
        path = os.path.join(img_dir, filename)
        if os.path.isdir(path):
            files_in_subdir += get_file_names(path, sample_rate=sample_rate, existing_names=existing_names)
        elif np.random.rand(1) <= sample_rate and filename[-1]=='f':
            file_names.append(os.path.join(img_dir, filename))
    file_names = [f for f in file_names if to_jpg_name(f) not in existing_names]
    file_names = file_names + files_in_subdir
    _logger.info("Found {} files in {}, including {} in sub folders.".format(len(file_names),
                                                                             img_dir,
                                                                             len(files_in_subdir)))
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
        jpg = to_jpg_name(tiff)

        reader = canon.TiffReader(canon.TiffReader.PILATUS)
        reader.loadtiff(tiff)
        reader.remove_background()
        reader.normalize()
        data = reader.image()
        if np.median(data) > 0 or np.mean(data) < 1e-3:
            print(jpg, np.median(data), np.max(data), np.mean(data))
            # continue
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
    import pandas as pd
    from canon.common.init import init_mpi_logging
    init_mpi_logging("logging_mpi.yaml")

    file_names = []
    input_dir = "/Volumes/G-DRIVE/BL1232_Oct2019/samp4-1-2_pillar"
    output_dir = "img/samp4-1-2_pillar_40_50"
    if MPI_RANK == 0:
        # existing_names = get_existing_names(["img/test_981"])
        existing_names = []
        file_names = get_file_names(input_dir, sample_rate=1, existing_names=existing_names)
    process_images(file_names, output_dir)

    # process_images(["img/test/au29_m1.tif"], "img/test")

    # reader = canon.TiffReader(canon.TiffReader.PILATUS)
    # # reader.loadtiff("img/test/NiTi_30C_00672.tif")
    # # reader.loadtiff("img/test/BTO_25C_wb3_05677.tif")
    # reader.loadtiff("img/test/au29_area_00068.tif")
    #
    # reader.remove_background()
    # reader.normalize()
    # img = reader.image()
    # # img = resize(img, (128, 128), mode='reflect')
    # img = Image.fromarray(img.astype(np.uint8))
    # img.save(os.path.join("img/test", "test00001.jpg"))







