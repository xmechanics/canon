import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def plot_seq(Z, step, colormap='gist_ncar', filename='untitled'):
    dir = os.path.dirname(os.path.abspath(__file__))
    save_directory = os.path.join(dir, '{:s}.png'.format(filename))
    x_step = step[0]
    y_step = step[1]
    # save_directory = 'img/au27_m{:s}.png'.format(key)
    font = {'weight': 'light', 'size': 10}
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(4, 3), dpi=150)
    gs = matplotlib.gridspec.GridSpec(1, 1)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    ax = plt.subplot(gs[0])
    ax.set_ylabel('Y ({:d} um/px)'.format(y_step))
    ax.set_xlabel('X ({:d} um/px)'.format(x_step))

    cmap = plt.get_cmap(colormap)
    cmap.set_bad(color='k', alpha=None)
    Z_mask = np.ma.array(Z, mask=np.isnan(Z))
    ax.imshow(Z[::-1, ::], interpolation='none', cmap=cmap, aspect=y_step / x_step, vmin=np.min(Z_mask), vmax=np.max(Z_mask))
    plt.savefig(save_directory, bbox_inches='tight', dpi=150)


def plot_contourf(Z, step, colormap='gist_ncar', filename='untitled'):
    dir = os.path.dirname(os.path.abspath(__file__))
    save_directory = os.path.join(dir, 'img/{:s}.png'.format(filename))
    X, Y = np.meshgrid(xrange(len(Z[0, :])), xrange(len(Z[:, 0])))

    x_step = step[0]
    y_step = step[1]
    # save_directory = 'img/au27_m{:s}.png'.format(key)
    font = {'weight': 'light', 'size': 10}
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(4, 3), dpi=150)
    gs = matplotlib.gridspec.GridSpec(1, 1)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    ax = plt.subplot(gs[0])
    ax.set_ylabel('Y ({:d} um/px)'.format(y_step))
    ax.set_xlabel('X ({:d} um/px)'.format(x_step))
    num_levels = 10
    zmin = np.min(Z)
    zmax = np.max(Z)
    levels = np.arange(zmin - 1, zmax + 1, (zmax - zmin + 2) / num_levels)
    origin = 'lower'
    CS3 = ax.contourf(X, Y, Z, levels,
                      colors=('r', 'g', 'b'),
                      origin=origin,
                      extend='both')
    CS3.cmap.set_under('yellow')
    CS3.cmap.set_over('cyan')
    CS4 = ax.contour(X, Y, Z, levels,
                     colors=('k',),
                     linewidths=(1,),
                     origin=origin)
    plt.clabel(CS4, fmt='%2.2f', colors='w', fontsize=6)
    plt.savefig(save_directory, bbox_inches='tight', dpi=150)


def get_imgn(nx, ny, numofcol):
    imgn = ny * numofcol + nx + 1
    return imgn


