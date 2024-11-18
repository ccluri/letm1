import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.ticker import LogLocator, NullFormatter


plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',
    'xtick.labelsize': 6,
    'xtick.major.size': 2,
    'xtick.major.width': 0.5,
    'xtick.major.pad': 1,
    'xtick.minor.size': 1,
    'xtick.minor.width': 0.5,
    'ytick.labelsize': 6,
    'ytick.major.size': 2,
    'ytick.major.width': 0.5,
    'ytick.major.pad': 1,
    'ytick.minor.size': 1,
    'ytick.minor.width': 0.5,
    'font.size': 7,
    'axes.labelsize': 7,
    'axes.labelpad': 1,
    'axes.titlesize': 7,
    'axes.titlepad': 2,
    'legend.fontsize': 7,
    'axes.linewidth': 0.5
})


def add_arrow(line, position=None, direction='right', size=15,
              color=None, arrowstyle='->', num=1):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   (x,y)-position of the arrow. If None, min*1.007 of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()
    
    if position is None:
        position = np.min(xdata)
    # find closest index
    # start_ind = np.argmin(np.absolute(xdata - position))  # - 1500
    start_ind = np.argmin(np.linalg.norm(np.stack((xdata, ydata)) -
                                         np.array(position).reshape(2, 1),
                                         axis=0))
    print('Verify that this has changed since, offset?')
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1
    for ii in range(num):
        strt_ = start_ind + int(ii*750)
        end_ = end_ind + int(ii*750)
        line.axes.annotate('',
                           xytext=(xdata[strt_], ydata[strt_]),
                           xy=(xdata[end_], ydata[end_]),
                           arrowprops=dict(arrowstyle=arrowstyle, color=color),
                           size=size)


def cm_to_inches(vals):
    return [0.393701*ii for ii in vals]


def add_sizebar(ax, size, loc=4, size_vertical=2, text=None):
    if not text:
        text = str(size) + ' sec'
    asb = AnchoredSizeBar(ax.transData,
                          int(size),
                          text,
                          loc=loc,
                          pad=0.5, borderpad=.4, sep=5,
                          frameon=False,
                          size_vertical=size_vertical)
    ax.add_artist(asb)
    return ax


def align_axis_labels(ax_list, axis='x', value=-0.25):
    for ax in ax_list:
        if axis == 'x':
            ax.get_xaxis().set_label_coords(0.5, value)
        else:
            ax.get_yaxis().set_label_coords(value, 0.5)


def add_logticks(axx, xax=True, ticks=[1, 10, 100, 1000]):
    if xax:
        axx.set_xticks(ticks)
        x_minor = LogLocator(base=10.0, subs=np.arange(1.0, 10.0)*0.1,
                             numticks=10)
        axx.xaxis.set_minor_locator(x_minor)
        axx.xaxis.set_minor_formatter(NullFormatter())
    else:
        axx.set_yticks(ticks)
        y_minor = LogLocator(base=10.0, subs=np.arange(1.0, 10.0)*0.1,
                             numticks=10)
        axx.yaxis.set_minor_locator(y_minor)
        axx.yaxis.set_minor_formatter(NullFormatter())
    return axx


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def remove_spines(axs, bottom=False, left=False):
    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if bottom:
            ax.spines['bottom'].set_visible(False)
        if left:
            ax.spines['left'].set_visible(False)
    return

def remove_ticks_labels(axs):
    for ax in axs:
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
