#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid.inset_locator import inset_axes, mark_inset

sns.set_style('whitegrid', {'grid.linestyle': '--'})
#sns.set_style('ticks')
sns.set_context("poster")

numLC     = 2
labelsize = 35
fontsize  = 25
legend_fontsize = 20

plt.rc('font',       size=fontsize) # controls default text sizes
plt.rc('axes',  titlesize=fontsize) # fontsize of the axes title
plt.rc('axes',  labelsize=fontsize) # fontsize of the x and y labels
plt.rc('xtick', labelsize=fontsize) # fontsize of the tick labels
plt.rc('ytick', labelsize=fontsize) # fontsize of the tick labels
plt.rc('legend', fontsize=fontsize) # legend fontsize
plt.rc('figure',titlesize=fontsize) # fontsize of the figure title

label = ['CM', 'BoB', 'FCHL19']
colors = ['C0', 'C1', 'C2']
markers = ['o', 'd', 's']

def read_data(f):
  real = np.asarray([])
  est  = np.asarray([])

  lines = open(f, 'r').readlines()

  for line in lines:
    tokens = line.split()
    Ereal  = float(tokens[1])
    Eest   = float(tokens[2])

    real = np.append(real, Ereal)
    est  = np.append(est, Eest)

  return real, est


def scatter(real, est, axin, color_inset, label_inset):

  #sns.scatterplot(est, real, color=color_inset, label=label_inset, ax=axin)
  x = np.linspace(0, 60, 100)
  y = np.linspace(0, 60, 100)
  sns.lineplot(x=x, y=y, color='k', linestyle='--', ax=axin, linewidth=1.5)
  axin.lines[0].set_linestyle("--")
  sns.scatterplot(x=est, y=real, color=color_inset, ax=axin, s=10)

  axin.set_xlim([0, 60])
  axin.set_ylim([0, 60])

  axin.set_xlabel(r'$E_{\mathrm{a}}^{\mathrm{est}}$' , labelpad=-10)
  axin.set_ylabel(r'$E_{\mathrm{a}}^{\mathrm{ref}}$', labelpad=-10)



  axin.spines['right'].set_color('none')
  axin.spines['top'].set_color('none')
  axin.spines['bottom'].set_position(('axes', -0.05))
  axin.spines['bottom'].set_color('black')
  axin.spines['left'].set_color('black')
  axin.yaxis.set_ticks_position('left')
  axin.xaxis.set_ticks_position('bottom')
  axin.spines['left'].set_position(('axes', -0.05))



def get_inset_data():
  fe2  = 'data/scatter_e2.txt'
  fsn2 = 'data/scatter_sn2.txt'

  e2_real, e2_est   = read_data(fe2)
  sn2_real, sn2_est = read_data(fsn2)

  return e2_real, e2_est, sn2_real, sn2_est

#  fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 6))

#  scatter(e2_real, e2_est, axes[0], 'E2', 'C4')
#  scatter(sn2_real, sn2_est, axes[1], r'S$_N$2', 'C5')


def get_data(filename):
  lines = open(filename, 'r').readlines()

  N = np.array([])
  energies = np.array([])

  for line in lines:
    tokens = line.split()
    N = np.append(N, int(tokens[2]))
    energies = np.append(energies, float(tokens[3]))

  return N, energies

def plot_lc(N, energies, label, ax, color, marker):
  N = np.log(N)
  energies = np.log(energies)

  sns.scatterplot(x=N, y=energies, color=color, marker=marker, s=200, ax=ax)
  sns.regplot(x=N, y=energies, ci=0, color=color, marker=marker,  line_kws={'linewidth': '2'}, ax=ax)
#  leg = ax.legend(fontsize=legend_fontsize)

#  ax.set_ylim([np.log(2.0),np.log(5.0)])

#  if i == 0:
#    axin = inset_axes(ax, "50%", "30%", bbox_to_anchor=(.18, .25, .8, .8), bbox_transform=ax.transAxes, loc=1)
#    scatter(real, est, axin, 'C5', 'E2')
#  if i == 5:
#    axin = inset_axes(ax, "50%", "30%", bbox_to_anchor=(.18, .25, .8, .8), bbox_transform=ax.transAxes, loc=1)
#    scatter(real, est, axin, 'C5', r'S$_N$2')


def set_ticks(ax):
  ax.set_xticks(np.array([np.log(625), np.log(1250), np.log(2500), np.log(5000)]))
  ax.set_xticklabels(['625', '1250', '2500','5000'])
  ax.set_yticks(np.array([np.log(0.5), np.log(1.0), np.log(2.0), np.log(4.0), np.log(6)]))
  ax.set_yticklabels(['0.5', '1.0', '2.0', '4.0', '6.0'])
#  ax.tick_params(labelsize=labelsize)

  ax.spines['right'].set_color('none')
  ax.spines['top'].set_color('none')
  ax.spines['bottom'].set_position(('axes', -0.05))
  ax.spines['bottom'].set_color('black')
  ax.spines['left'].set_color('black')
  ax.yaxis.set_ticks_position('left')
  ax.xaxis.set_ticks_position('bottom')
  ax.spines['left'].set_position(('axes', -0.05))


def set_labels(ax):
#  ax.set_title(title, fontsize=fontsize)
  ax.set_xlabel('$N$')#, fontsize=fontsize)
  ax.set_ylabel('MAE [kcal$\cdot$mol$^{-1}$]')#, fontsize=fontsize)
#  axes[1][0].set_ylabel('MAE [kcal$\cdot$mol$^{-1}$]'#), fontsize=fontsize)


def set_acc(ax):
#  ax.axhline(np.log(1.0), color='C9', ls='--')
  ax.axhline(np.log(1.0), color='C9', ls='--')
#
  ax.text(np.log(1500.), np.log(1.05), 'Chem acc.')#, fontsize=35)
#  ax.text(np.log(200.), np.log(1.), 'Chem acc.',fontsize=20)

def set_axis(ax):
  ax.spines['right'].set_color('none')
  ax.spines['top'].set_color('none')
  ax.spines['bottom'].set_position(('axes', -0.05))
  ax.spines['bottom'].set_color('black')
  ax.spines['left'].set_color('black')
  ax.yaxis.set_ticks_position('left')
  ax.xaxis.set_ticks_position('bottom')
  ax.spines['left'].set_position(('axes', -0.05))

  ax.grid(False)
  ax.set_xticks([])
  ax.set_yticks([])

  ax.set_xlabel(r'Log($N_{train}$)')
  ax.set_ylabel(r'Log(MAE)')

def ficticious_figure(ax):
  set_axis(ax)


  ax.axhline(0.2, color='gray', ls='--')
  ax.axvline(0.6, color='gray', ls='-.')
  ax.axvline(0.8, color='gray', ls=':')

  x1, y1 = [0.,0.8], [0.6, 0.2]
  x2, y2 = [0.,0.4], [0.7, 0.5]
  x3, y3 = [0.4,0.8], [0.5, 0.4]

  ax.plot(x1, y1, color='C3')
  ax.plot(x2, y2, color='C4')
  ax.plot(x3, y3, color='C4')

  ax.text(x=0.05, y=0.15, s="Target accuracy")
  ax.text(x=0.05, y=0.35, s="Good model")
  ax.text(x=0.05, y=0.70, s="Bad model")

  ax.text(x=0.62, y=0.5, s="Training data", rotation=90)
  ax.text(x=0.82, y=0.27, s="Estimated data needed", rotation=90)

  ax.set_xlim(0,1)
  ax.set_ylim(0.1,0.8)

if __name__ == '__main__':

  filenames = [ "data/CM.txt", "data/BoB.txt", "data/FCHL.txt" ]

  #f, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(20, 8))
  f, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
#  e2_real, e2_est, sn2_real, sn2_est = get_inset_data()

  ficticious_figure(ax[0])

  for i, filename in enumerate(filenames):
    N, energies = get_data(filename)

    plot_lc(N, energies, label[i], ax[1], colors[i], markers[i])
    set_ticks(ax[1])
    set_labels(ax[1])
    set_acc(ax[1])

    ax[1].set_xlim(np.log(600),np.log(5300))

  f.subplots_adjust(top=0.8, bottom=0.2, wspace = 0.5)

  legend_elements = [ Line2D([0], [0], marker=markers[0], color=colors[0], label=label[0]),
                      Line2D([0], [0], marker=markers[1], color=colors[1], label=label[1]),
                      Line2D([0], [0], marker=markers[2], color=colors[2], label=label[2]),]

  f.legend(handles=legend_elements, loc='upper center', ncol=4, bbox_to_anchor=(0.50, 1.04),frameon=False)

  plt.tight_layout()

  f.savefig("figs/lc.png")
  f.savefig("figs/lc.pdf")
  #plt.show()
