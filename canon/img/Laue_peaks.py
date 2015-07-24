import numpy as np
x0 = 539.689
y0 = 197.444
def read_peaks(filename):
    f = open(filename,'r')
    x = []
    y = []
    flag = 0
    inten = []
    for line in f:
        flag+=1
        words = line.split()
        if flag == 2:
            cap = [words.index('x'), words.index('y'), words.index('integr(obs)')]
        if flag > 2:
            x.append(float(words[cap[0]]))
            y.append(-float(words[cap[1]]))
            inten.append(float(words[cap[2]]))
    i_max = np.max(np.array(inten))
    inten = inten/i_max
    f.close()
    return np.array([x, y, inten]).T
def flt_peaks(peak_list, threshold):
    flt_idx = [i for i, v in enumerate(peak_list[:,2]) if v > threshold]
    return np.array([peak_list[i][0:2] for i in flt_idx])






aust = flt_peaks(read_peaks('/Users/sherrychen/project/xmas-als/xmas_web/m4fine_m00465.DAT'), 0.01)

import matplotlib
from matplotlib import pyplot as plt
font = {'weight' : 'light',
        'size'   : 6}
matplotlib.rc('font', **font)
fig = plt.figure(figsize=(3, 3), dpi = 600)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
# mart = flt_peaks(read_peaks('data/mart_v4.txt'), 0.15)
# plt.scatter(mart[:,0], mart[:,1], s=8, c='r', lw=1)
# mart = flt_peaks(read_peaks('data/mart_v3.txt'), 0.1)
# plt.scatter(mart[:,0], mart[:,1], s=8, c='b', lw=1, alpha=0.6)
# mart = flt_peaks(read_peaks('data/mart_v3.txt'), 0.1)
# plt.scatter(mart[:,0], mart[:,1], s=8, c='g', lw=1, alpha=0.3)
# mart = flt_peaks(read_peaks('data/mart_v5.txt'), 0.08)
# plt.scatter(mart[:,0], mart[:,1], s=8, c='c', lw=1, alpha=0.6)

plt.scatter(aust[:,0], aust[:,1], s=10, c='r', lw=0.01, alpha=0.6)

w = 1043
h = 981
plt.xlim([0, w])
plt.ylim([-h, 0])
plt.axes().set_aspect('equal')
plt.xlabel('position along X (pixels)')
plt.ylabel('position along Y (pixels)')
plt.savefig('peaks_img/m4fine_465peaks.png', bbox_inches="tight", dpi=300)
# plt.show()
plt.close()