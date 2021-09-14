#!/usr/bin/env python3

import sys
import random
from datetime import datetime
import numpy as np
from copy import deepcopy
import qml
from qml.kernels import get_local_symmetric_kernels
from qml.kernels import get_local_kernels
from qml.representations import generate_fchl_acsf
from qml.math import cho_solve
import itertools
from time import time

# Function to parse datafile to a dictionary
def get_energies(filename):
  """ Returns a dictionary with heats of formation for each xyz-file.
  """

  f = open(filename, "r")
  lines = f.readlines()
  f.close()

  energies = dict()

  for line in lines:
    tokens = line.split()

    xyz_name = tokens[0]
    hof = float(tokens[1])*627.509

#    if hof < 100 and hof > 0: energies[xyz_name] = hof
    energies[xyz_name] = hof

  return energies


if __name__ == "__main__":
  # Parse file containing PBE0/def2-TZVP heats of formation and xyz filenames
  data  = get_energies("train.txt")
  data2 = get_energies( "test.txt")

  # Generate a list of fml.Molecule() objects
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

  N   = [625, 1250, 2500, 5000]
  nModels  = 10
  total   = len(mols)
  sigma = [0.1 * 2**i for i in range(15)]
  ll = [1e-7, 1e-9, 1e-11]

  x = []
  q = []

  fancy_element_list_because_anders_wants_it_that_way = [1,6,7,8,9]

  print("\n -> generating the representation")
  start = time()
  # Generate descriptor for each molecule
  for mol in mols:
    x1 = generate_fchl_acsf(mol.nuclear_charges, mol.coordinates, gradients=False, pad=19, elements=fancy_element_list_because_anders_wants_it_that_way)
    x.append(x1)
    q.append(mol.nuclear_charges)

  X    = np.array(x)
  Q    = q

  x_test = []
  q_test = []

  for mol in mols_test:
    x1 = generate_fchl_acsf(mol.nuclear_charges, mol.coordinates, gradients=False, pad=21, elements=fancy_element_list_because_anders_wants_it_that_way)
    x_test.append(x1)
    q_test.append(mol.nuclear_charges)

  X_test = np.array(x_test)
  Q_test = q_test

  end = time()
  print(str(end-start))


  print("\n -> calculating the Kernel ")
  start = time()
  K      = get_local_symmetric_kernels(X, Q, sigma)
  K_test = get_local_kernels(X_test, X, Q_test, Q, sigma)


  end = time()

  Yprime = np.asarray([ mol.properties for mol in mols ])
  Y_test = np.asarray([ mol.properties for mol in mols_test ])

  random.seed(667)

  for j in range(len(sigma)):
    print('\n')
    for l in ll:
      print()
      for train in N:
        maes = []
        for i in range(nModels):
          split = list(range(total))
          random.shuffle(split)

          training_index = split[:train]

          Y = Yprime[training_index]

          C = deepcopy(K[j][training_index][:,training_index])
          C[np.diag_indices_from(C)] += l

          alpha = cho_solve(C, Y)

          Yss = np.dot((K_test[j][training_index]).T, alpha)
          diff = Yss  - Y_test
          mae = np.mean(np.abs(diff))
          maes.append(mae)

        print(str(l) + "\t" + str(sigma[j]) +  "\t" + str(train) + "\t" + str(sum(maes)/len(maes)))
