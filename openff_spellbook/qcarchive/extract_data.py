import pickle
import collections
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
import numpy as np

import openff-spellbook.tools.const as const

with open('oFF-1.1.0.p', 'rb') as fid:
    print("Loading oFF data")
    oFF10 = pickle.load(fid)
with open('QCA.p', 'rb') as fid:
    print("loading QCA data")
    QCA = pickle.load(fid)
if QCA.db is None:
    with open('QCA.db.p', 'rb') as fid:
        print("loading QCA db")
        QCA.db = pickle.load(fid).db


converters = { "Bonds" : const.bohr2angstrom, \
               "Angles": 1.0, \
               "ImproperTorsions": 1.0, \
               "ProperTorsions": 1.0, \
               "vdW": 1.0 \
}
data_chart = { "Bonds" : "bonds.p", \
               "Angles": "angles.p", \
               "ImproperTorsions": "outofplane.p", \
               "ProperTorsions": "torsions.p" \
}

label_db = oFF10.db.get( "ROOT").get( "data")

param_types = list(data_chart.keys())

# get the last structure from opts -> min energy 
mol_idx = -1

for param in param_types:
    data = None
    filename = data_chart.get( param)
    with open( filename, 'rb') as fid:
#        print("Loading", filename)
        data = pickle.load(fid)
    for entry in QCA.iter_entry():
        labels = oFF10.db.get( entry.payload).get( "data").get( param)
        mol_name = QCA.db.get( entry.payload).get( "entry").name
        for group in labels:
            label = labels.get( group)
            d = collections.defaultdict(list)
            for cons in QCA.node_iter_depth_first(entry, select="Constraint"):
                ang = eval(cons.payload)[0]
                for opt in QCA.node_iter_depth_first( cons, select="Optimization"):
                    ene = QCA.db.get( opt.payload).get( "data").get( "energies")[ mol_idx]
                    # assume we are taking only minimum energy
                    grad = QCA.node_index.get( opt.children[ mol_idx])
                    mol = QCA.node_index.get( grad.children[0])
                    syms = QCA.db.get( mol.payload).get( "data").get( "symbols")
                    vals = data.db.get( mol.payload).get( group) 
                    vals *= converters.get( param)
                    out_str = "{:12s} {:16.10e} {:8.2f} {:4s} {:20s} {:10.4f} {:64s}"
                    print(out_str.format(entry.payload, ene, ang, label, \
                          "-".join([str(syms[x-1])+str(x) for x in group]), vals[0], \
                          mol_name ))

#                    # sigh, assume vals are scalars
#                    d[ang].append( [ene, vals[0]] )
