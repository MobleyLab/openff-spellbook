import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import collections
import logging
import pickle
import simtk.unit
import collections

FORMAT = '%(levelname)-10s %(asctime)-15s %(funcName)-20s %(lineno)-s %(message)s'
logging.basicConfig(format=FORMAT)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.NOTSET)

def main():
    with open(sys.argv[1], 'rb') as fid:
        d = pickle.load(fid)
    mols_combined = combine(d)
    search = {"Mol": mols_combined, "Bonds": [], "Angles": [], "ImproperTorsions": [], "ProperTorsions": [], "Energy": collections.OrderedDict()}
    for p in sys.argv[2:]:
        if(p[0] == 'a'):
            search['Angles'].append(p) 
        elif(p[0] == 'b'):
            search['Bonds'].append(p) 
        elif(p[0] == 'i'):
            search['ImproperTorsions'].append(p) 
        elif(p[0] == 't'):
            search['ProperTorsions'].append(p)
    #search['Energy']['qm'] = None
    #search['Energy']['oFF'] = ['epot', 'vdW', 'Bonds', 'Angles', 'ProperTorsions']
    print("Searching", sum([1 for y in search["Mol"]]), "molecules with", sum([1 for y in search["Mol"] for x in y]), "conformations")
    print("Query:", [(x,v) for (x,v) in search.items() if x != "Mol"], end="\n\n")
    flag_measurements(mindb=d, groups=mols_combined, params=search)

if(__name__ == "__main__"):
    main()

