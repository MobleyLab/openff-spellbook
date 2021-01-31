#!/usr/bin/env python3 

import itertools
import logging
import numpy as np

def flatten_list(l, times=1):
    if times == 0: return l
    if times == -1:
        if isinstance(l[0], list):
            ll = [a for b in l if hasattr(b,"__iter__") for a in b]
            return flatten_list(ll, times)
        else:
            return l
    else:
        return flatten_list(
                 [a for b in l if hasattr(b,"__iter__") for a in b], times-1)

def argsort_labels(l):
    letter = [i[0] for i in l]
    num = [int(i[1:]) for i in l]
    return np.lexsort((num, letter))

def strip_conformation_number(mol):
    ret = mol.split("-")
    if(len(ret) == 1):
        ret = mol
    else:
        try:
            isnumber = int(ret[-1])
            ret = "".join(ret[:-1])
        except ValueError:
            ret = mol
    return ret

def get_conformation_number(mol):
    ret = mol.split("-")
    if(len(ret) == 1):
        ret = "1"
    else:
        try:
            isnumber = int(ret[-1])
            ret = isnumber
        except ValueError:
            ret = "1"
    return ret

def load_xyz(xyz_file):
    
    fid = open(xyz_file)
    xyz = []
    atoms = int(fid.readline())

    while True:

        fid.readline()
        xyz_frame = [list(map(float, fid.readline().split()[1:])) for i in range(atoms)]
        xyz.append(xyz_frame)
        n = fid.readline()
        if n == "":
            break

    fid.close()

    return np.array(xyz)
