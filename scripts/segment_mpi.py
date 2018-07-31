import os
import sys
import logging
from timeit import default_timer as timer
from itertools import groupby
from mpi4py import MPI
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

MPI_COMM = MPI.COMM_WORLD
MPI_RANK = MPI_COMM.Get_rank()

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
_logger = logging.getLogger(__name__)

from canon.pattern import LatentExtractor, PeakNumberExtractor, CombinedExtractor
from canon.dat.datreader import read_dats, idx2XY
from canon.pattern.model import GMModel, KMeansModel, BGMModel, MeanShiftModel
from canon.pattern.labeler import SeqLabeler
from canon.util import split_workload
from canon.mpi import extract_sample_features, score_dir


def plot_seq(ax, Z, step, colormap='gist_ncar', filename='untitled'):
    x_step = step[0]
    y_step = step[1]
    ax.set_ylabel(r'Y ({:d} $\mu$m/px)'.format(y_step))
    ax.set_xlabel(r'X ({:d} $\mu$m/px)'.format(x_step))
    cmap = plt.get_cmap(colormap)
    cmap.set_bad(color='k', alpha=None)
    Z_mask = np.ma.array(Z, mask=np.isnan(Z))
    # ax.imshow(Z[::-1, ::], interpolation='none', cmap=cmap, aspect=y_step / x_step, vmin=np.min(Z_mask), vmax=np.max(Z_mask))
    print(Z.shape)
    ax.imshow(Z[::-1, :, :], aspect=y_step / x_step)


if __name__ == '__main__':
    from canon.common.init import init_mpi_logging
    init_mpi_logging("logging_mpi.yaml")

    extractor1 = PeakNumberExtractor()
    extractor2 = LatentExtractor("ae_128_256_conv_4")
    extractor = CombinedExtractor([extractor1, extractor2])
    extractor = extractor2

    # CuAlNi_mart2
    # scratch = "/Users/sherrychen/scratch/"
    scratch = os.path.dirname(os.path.abspath(__file__))
    z_file = "Z.txt"
    z_plot = "Z"

    # tiff_dir = os.path.join(scratch, "img", "C_2_1_test_processed")
    # seq_files = [os.path.join(scratch, "seq", "C_2_1_test_.SEQ")]
    # NX = 25
    # NY = 20
    # sample_rate = 1.0

    # tiff_dir = os.path.join(scratch, "img", "CuAlNi_mart2_processed")
    # seq_files = [os.path.join(scratch, "seq", "CuAlNi_mart2_.SEQ")]
    # NX = 100
    # NY = 80
    # sample_rate = 1.0

    # tiff_dir = os.path.join(scratch, "img", "BTO_25C_wb3_processed")
    # seq_files = [os.path.join(scratch, "seq", "CuAlNi_mart2_.SEQ")]
    # NX = 100
    # NY = 60
    # sample_rate = 1.0

    # tiff_dir = os.path.join(scratch, "img", "C_2_1_mscan1")
    # seq_files = [os.path.join(scratch, "seq", "CuAlNi_mart2_.SEQ")]
    # NX = 60
    # NY = 59
    # sample_rate = 1.0

    tiff_dir = os.path.join(scratch, "img", "au30_mart4_fine")
    seq_files = [os.path.join(scratch, "seq", "CuAlNi_mart2_.SEQ")]
    NX = 100
    NY = 25
    sample_rate = 1.0

    # tiff_dir = os.path.join(scratch, "img", "C_4_2_scan1")
    # seq_files = [os.path.join(scratch, "seq", "CuAlNi_mart2_.SEQ")]
    # NX = 40
    # NY = 20
    # sample_rate = 1.0

    # tiff_dir = os.path.join(scratch, "img", "ZrO2_770C_wb1_processed")
    # seq_files = [os.path.join(scratch, "seq", "CuAlNi_mart2_.SEQ")]
    # NX = 110
    # NY = 80
    # sample_rate = 1.0

    step = (4, 10)
    # step = (5, 5)
    training_set = extract_sample_features(extractor, tiff_dir, sample_rate=sample_rate)    # sample_patterns on lives on core-0

    if MPI_RANK == 0:
        model = BGMModel()
        training_set = np.array(training_set)
        model.train(training_set, n_clusters=200, preprocessors=[])
        silhouette = model.compute_silhouette_score(training_set)
        calinski = model.compute_calinski_harabaz_score(training_set)
        _logger.info("Silhouette Score = {}, Calinski-Harabaz Score = {}".format(silhouette, calinski))
    else:
        model = None
    model = MPI_COMM.bcast(model, root=0)

    score_inds = score_dir(extractor, model, tiff_dir, limit=None, batch_size=200)

    # # only support relabeling Guassian Mixture based model for now
    # if MPI_RANK == 0:
    #     labeler = SeqLabeler(seq_files)
    # else:
    #     labeler = None
    # labeler = MPI_COMM.bcast(labeler, root=0)
    # score_inds = relabel(labeler, score_inds)

    if MPI_RANK == 0:
        Z = np.empty([NY, NX])
        Z[:] = np.nan
        for score, idx in score_inds:
            ix, iy = idx2XY(idx, NX)
            if score is not None:
                if ix < NY:
                    if isinstance(score, tuple) or isinstance(score, list) or isinstance(score, np.ndarray):
                        Z[ix, iy] = score[0]
                    else:
                        Z[ix, iy] = score
                    if Z[ix, iy] is None:
                        print("{}, {}, Found None".format(ix, iy))
            # if Z[ix, iy] == np.nan:
            #     print("{}, {}, Found nan".format(ix, iy))
        _logger.info('Z matrix has %d nans' % sum(1 for row in Z for z in row if np.isnan(z)))
        np.savetxt(z_file, Z)
        _logger.info('Write Z matrix into ' + z_file + ' in ' + os.path.dirname(os.path.abspath(__file__)))

    # Visualization
    if MPI_RANK == 0:
        rcParams['font.size'] = 10
        rcParams['font.family'] = 'serif'
        rcParams['font.sans-serif'] = ['Times New Roman']
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 4))

        # fig = plt.figure(figsize=(4, 3), dpi=150)
        # gs = matplotlib.gridspec.GridSpec(1, 1)
        # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        Z = np.loadtxt(z_file)
        scaler0 = model.get_label_scaler(0)
        scaler1 = model.get_label_scaler(1)
        scaler2 = model.get_label_scaler(2)
        Z2 = np.zeros((Z.shape[0], Z.shape[1], 3))
        Z2[:, :, 0] = scaler0(Z)
        Z2[:, :, 1] = scaler1(Z)
        Z2[:, :, 2] = scaler2(Z)
        plot_seq(ax, Z2, step, colormap='jet', filename=os.path.join(scratch, "img", z_plot))
        # plt.tight_layout()
        fig.savefig("img/Z.pdf", bbox_inches='tight', dpi=300)

