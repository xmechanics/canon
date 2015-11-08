import numpy as np
from plotseq import plot_seq

Z = np.loadtxt('Z.txt')
plot_seq(Z, (5, 5), colormap='jet', filename="clustering_quartz500Mpa")
