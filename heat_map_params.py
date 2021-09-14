#! /usr/bin/env python3

import sys
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.interpolate import Rbf

import pandas as pd
import numpy as np

cmap = "YlGnBu"

fontsize = 20
plt.rc('font',       size=fontsize) # controls default text sizes
plt.rc('axes',  titlesize=fontsize) # fontsize of the axes title
plt.rc('axes',  labelsize=fontsize) # fontsize of the x and y label
plt.rc('xtick', labelsize=fontsize) # fontsize of the tick labels
plt.rc('ytick', labelsize=fontsize) # fontsize of the tick labels
plt.rc('legend', fontsize=fontsize) # legend fontsize
plt.rc('figure',titlesize=fontsize) # fontsize of the figure title

def make_axis(ax):
  ax.spines['right'].set_color('none')
  ax.spines['top'].set_color('none')
  ax.spines['bottom'].set_position(('axes', -0.05))
  ax.spines['bottom'].set_color('black')
  ax.spines['left'].set_color('black')
  ax.yaxis.set_ticks_position('left')
  ax.xaxis.set_ticks_position('bottom')
  ax.spines['left'].set_position(('axes', -0.05))

def read_data(f):
  colnames=['lambda', 'sigma', 'MAE']
  return pd.read_csv(f, names=colnames, header=None)
#
def heat_map(df, fout):

  fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12,6))

  heatmap_data   = pd.pivot_table(df, values='MAE',   index=['lambda'], columns='sigma')
#  heatmap_data_f = pd.pivot_table(df, values='MAE_f', index=['lambda'], columns='sigma')
  sns.heatmap(heatmap_data, ax=ax, cmap=cmap, annot=True, fmt='.2g', robust=True, cbar_kws={'label': r'MAE $[\mathrm{kcal}\cdot \mathrm{mol}^{-1}$]'})
#  sns.heatmap(heatmap_data_f, ax=ax[1], cmap=cmap, annot=True, fmt='.2g', robust=True, cbar_kws={'label': r'Forces $[kcal\cdot mol^{-1}\AA^{-1}$]'})


  plt.tight_layout()

  fig.savefig("figs/" + fout)

if __name__ == "__main__":

  filename = sys.argv[1]
  fout = filename[:-4] + ".png"
  df = read_data(filename)

  heat_map(df, fout)


