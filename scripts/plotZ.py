import numpy as np
from .plotseq import plot_seq

Z = np.loadtxt('Z.txt')
plot_seq(Z, (2, 2), colormap='jet', filename="img/clustering_au30_area_2k")
