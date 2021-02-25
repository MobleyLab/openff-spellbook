#!/usr/bin/env python3

# Imports
#{{{
import pickle
import collections
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
import numpy as np
import re
import os

from offsb.op.geometry import ( \
        BondOperation, 
        AngleOperation,
        ImproperTorsionOperation,
        TorsionOperation)

import offsb.tools.const as const
import simtk.unit as unit
import offsb.qcarchive as qca

PREFIX="."

#}}}


################################################################################
# The purpose of this script is to take the data from QCArchive, and join
# it with a calculation done on top of it. This can be whatever, but the
# example shown here is the OpenMM energy calculation produced from the
# other example
# Ideally it should work with any other analysis as well, as long as it was
# done using a TreeOperation (see offsb/op for available operations).
################################################################################

# Misc utils
#{{{

def tuple_to_hyphenated( x):
    if isinstance( x, tuple):
        return re.sub("[() ]", "", str(x).replace(",","-"))
    else:
        return x


# Converters; really this is only needed for distances (bonds), since QCA uses
# bohrs/degrees
converters = { \
#               "Bonds" : const.bohr2angstrom, \
#               "Angles": 1.0, \
#               "ImproperTorsions": 1.0, \
#               "ProperTorsions": 1.0, \
#               "vdW": 1.0 \
               "MMEnergyA": 1.0, \
#               "MMEnergyB": 1.0 \
}

# Look up table of where to find the data
data_chart = { \
#               "Bonds" : "bonds.p", \
#               "Angles": "angles.p", \
#               "ImproperTorsions": "outofplane.p", \
#               "ProperTorsions": "torsions.p" \
               "MMEnergyA": "oMM.oFF-Parsley-uncons.p" \
#               "MMEnergyB": "oMM.oFF-Parsley-uncons"  \
}


#}}}

# Main function to parse data
#{{{
def extract_torsions_for_same_molecule(QCA, param_types, data_chart, converters):
    """ 
    Create a table-like file of a TorsionDrive set, and print the results
    of an auxiliary calculation, like OpenMM single points
    """

    # Assume to get the last structure from opts -> min energy 
    mol_idx = -1
    for param in param_types:
            data = None

            filename = data_chart[param]
            with open( os.path.join( PREFIX, filename), 'rb') as fid:
                data = pickle.load(fid)

            key = 'canonical_isomeric_explicit_hydrogen_mapped_smiles'
            for folder in QCA.combine_by_entry():

                # check if there are torsions, if not, skip (for now)
                # since we are print files based on some scanned value
                has_torsion = False
                for n in folder.children:
                    if QCA[n].name == "TorsionDrive":
                        has_torsion = True
                        break
                if not has_torsion:
                    continue

                print(folder)
                for n in folder.children:
                    constraints = set([c.payload[:2] for c in
                        QCA.node_iter_depth_first(QCA[n], select="Constraint")])

                # Start supporting multiple scanned values
                num_angs = len(constraints)
                out_str = "{:12s} {:12s} {:16.10e} "+"{:12s} {:12s} {:8.4f}"*num_angs +" {:10.4f} {:64s}\n"
                
                i = 0
                for entry_id in folder.children:
                    entry_node = QCA[entry_id]
                    print( "  ",i, entry_node)
                    i += 1

                    entry = QCA.db[entry_node.payload]["entry"]
                    mol_name = entry.name
                    smiles_indexed = entry.attributes[ key]

                    # choose between all mins or just min along a TD
                    # Note that not sure how TD iter will work when entry 
                    # is an optimization (it crashes right now)
                    #nodes = list(QCA.node_iter_optimization_minimum(entry_node, select="Molecule"))
                    nodes = list(QCA.node_iter_torsiondriverecord_minimum(entry_node, select="Molecule"))

                    # Sort the molecules by scanned value, e.g. TD angle
                    order = np.arange( len( nodes))
                    cons = []
                    for node in nodes:
                        val = tuple([ c.payload[2] for c in \
                            QCA.node_iter_to_root( node, 
                                select="Constraint")])
                        if len(val) > 0:
                            cons.append(val)

                    if len(cons) == 0:
                        nodes_in_order = nodes
                        order = np.arange(len(nodes))
                    else:
                        cons = np.array( cons)
                        order = np.lexsort( cons.T)
                        nodes_in_order = [nodes[i] for i in order]
                    cons_set = cons
                    fnm = entry_node.payload + "." + param + ".dat"
                    fd = open( fnm, 'w')
                    spec_written = False

                    # Assumes -1 is root, so -2 are the datasets
                    ds = list(QCA.node_iter_to_root( entry_node))[-2]
                    mols = []

                    # Finally, iterate through molecules, collecting the data
                    # and finally printing it out
                    for m_idx, mol_node in enumerate(nodes_in_order):

                        # Each mol belongs to one optimization
                        # Needed to get the spec, but probably can use the spec
                        # node now
                        opt = next(QCA.node_iter_to_root( mol_node, select="Optimization"))
                        if not spec_written:
                            qc_spec = QCA.db.get( opt.payload).get( "data").get( "qc_spec")
                            method = str( qc_spec.method)
                            basis  = str( qc_spec.basis)
                            fd.write( ds.name + "\n" + method + "\n" + basis + "\n")

                            header = "{:12s} {:12s} {:16s} "+"{:12s} {:12s} {:8s}"*num_angs +" {:10s} {:64s}\n"
                            fd.write(header.format( "# QCAProc", " QCAMol", " QMEne", 
                                *[" ScanType", " Atoms", " ScanVal"] * num_angs, \
                                  param, " SmilesMapped" ))
                            spec_written = True

                        if QCA.db.get( opt.payload).get( "data").get( "energies") is None:
                            #fd.close()
                            print("No energies")
                            continue

                        try:
                            ene = QCA.db.get( opt.payload).get( "data").get( "energies")[ mol_idx]
                        except TypeError:
                            print("No energies for this mol", mol_idx)
                            #fd.close()
                            continue


                        mol_id = mol_node.payload
                        # assume we are taking only minimum energy
                        syms = QCA.db.get( mol_node.payload).get( "data").get( "symbols")
                        qc_mol = QCA.db.get( mol_node.payload).get( "data")

                        # this is the "data" e.g. openMM energy eval
                        try:
                            vals = data.db.get( mol_node.payload).get( "data").get("energy")
                            vals *= converters.get( param)
                        except Exception:
                            #fd.close()
                            print("No aux data from", data.name, mol_node.payload)
                            continue

                        # For all of the scanned angles in this molecule
                        # iterate and print all values
                        # E.g. if there were two scans, for each mol print
                        # the value of each scan (noting that for each mol,
                        # only one was driven. Right now we don't have any
                        # explicit 2D scans
                        angle_str = []
                        for cons in constraints:
                            indices = tuple_to_hyphenated(cons[1])
                            angle = TorsionOperation.measure_praxeolitic_single(qc_mol, list(cons[1]))
                            if np.sign(angle) != np.sign(cons_set[order[m_idx]]):
                                angle *= -1.0
                            angle_str = [cons[0],indices,float(cons_set[order[m_idx]])]
                            angle_str = [cons[0],indices,angle]

                            # Debug stuff testing dihedral calculations
                            #qc_angle = qcelemental.models.molecule.Molecule.from_data(qc_mol).measure(list(cons[1]))
                            #angle_str = [cons[0],indices,float(qc_angle)]
                            #angle_str = [cons[0],indices,angle]

                            if type(vals) is unit.Quantity:
                                vals = vals.value_in_unit( vals.unit)
                            print("    ", entry_node.payload, mol_id, ene, *angle_str,vals)
                            fd.write(out_str.format( entry_node.payload, mol_id, ene, *angle_str, \
                                  vals, \
                                  smiles_indexed ))

                fd.close()

#}}}

# Driver (main)
#{{{
if __name__ == "__main__":

    # Load data from disk
    with open(os.path.join(PREFIX,'QCA.p'), 'rb') as fid:
        QCA = pickle.load(fid)
    if QCA.db is None:
        with open(os.path.join(PREFIX, 'QCA.db.p'), 'rb') as fid:
            QCA.db = pickle.load(fid).db

    param_types = list(data_chart.keys())
    extract_torsions_for_same_molecule(QCA, param_types, data_chart, converters)

#}}}
