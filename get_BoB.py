#!/usr/bin/env python3

import sys
import time
from datetime import datetime
import random
#import cPickle
import numpy as np
from copy import deepcopy
import qml
from qml.representations import *
from qml.kernels import gaussian_kernel
from qml.kernels import laplacian_kernel
from qml.math import cho_solve
import itertools
from time import time

def get_energies(filename):
  """ returns dic with energies for xyz files
  """
  f = open(filename, "r")
  lines = f.readlines()
  f.close()

  energies = dict()

  for line in lines:
    tokens = line.split()
    xyz_name = tokens[0]
    Ebind = float(tokens[1]) * 627.509
    energies[xyz_name] = Ebind

  return energies


if __name__ == "__main__":

  data    = get_energies("train.txt")
  data2   = get_energies( "test.txt")


  mols = []
  mols_test = []

  for xyz_file in sorted(data.keys()):
    mol = qml.Compound()
    mol.read_xyz("../xyz/" + xyz_file + ".xyz")
    mol.properties = data[xyz_file]
    mols.append(mol)

  for xyz_file in sorted(data2.keys()):
    mol = qml.Compound()
    mol.read_xyz("../xyz/" + xyz_file + ".xyz")
    mol.properties = data2[xyz_file]
    mols_test.append(mol)


  bags = {
    "H":  max([mol.atomtypes.count("H" ) for mol in mols+mols_test]),
    "C":  max([mol.atomtypes.count("C" ) for mol in mols+mols_test]),
    "N":  max([mol.atomtypes.count("N" ) for mol in mols+mols_test]),
    "O":  max([mol.atomtypes.count("O" ) for mol in mols+mols_test]),
    "F":  max([mol.atomtypes.count("F" ) for mol in mols+mols_test]),
    "Cl": max([mol.atomtypes.count("Cl") for mol in mols+mols_test]),
    "Br": max([mol.atomtypes.count("Br") for mol in mols+mols_test]),
  }

  for mol in mols:
    mol.generate_bob(asize=bags)
  for mol in mols_test:
    mol.generate_bob(asize=bags)

  #N = [50,100,200,400,800,1600]
  N = [625, 1250, 2500, 5000]
  total = len(mols)
  nModels = 10
  ll = [1e-5, 1e-7]
  sigma = [0.1*2**i for i in range(8,15)]


  X        = np.asarray([mol.representation for mol in mols])
  X_test   = np.asarray([mol.representation for mol in mols_test])

  Yprime = np.asarray([ mol.properties for mol in mols ])
  Y_test = np.asarray([ mol.properties for mol in mols_test ])

  random.seed(667)

  for j in range(len(sigma)):
    print('\n\n -> calculate kernels')
    K      = laplacian_kernel(X, X, sigma[j])
    K_test = laplacian_kernel(X, X_test, sigma[j])

    for l in ll:
      print()
      for train in N:
        maes = []
        for i in range(nModels):
          split = np.array(list(range(total)))
          random.shuffle(split)

          training_index  = split[:train]

          Y = Yprime[training_index]


          C = K[training_index[:,np.newaxis],training_index]
          C[np.diag_indices_from(C)] += l
          alpha = cho_solve(C, Y)

          Yss = np.dot((K_test[training_index]).T, alpha)
          diff = Yss  - Y_test
          mae = np.mean(np.abs(diff))
          #mae = np.abs(diff)
          maes.append(mae)
          s = np.std(maes)/np.sqrt(nModels)
        print(str(l) + "\t" + str(sigma[j]) +  "\t" + str(train) + "\t" + str(sum(maes)/len(maes)) + " " + str(s))
        #print(str(l) + "\t" + str(sigma[j]) +  "\t" + str(train) + "\t" + maes + " " + str(s))
