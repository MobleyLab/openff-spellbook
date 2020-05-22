#!/usr/bin/env python3


# Imports
#{{{
import os
import qcfractal.interface as ptl
import offsb.qcarchive.qcatree as qca
import treedi.node as Node
import pandas as pd
import pickle
import qcelemental
#}}}


# Initialize and configure which datasets to retreive
#{{{
client = ptl.FractalClient()
with pd.option_context(
        'display.max_rows', 10000000, 
        'display.max_columns', 100,
        'display.max_colwidth', 10000):
    print(client.list_collections())

QCA = qca.QCATree( "QCA", root_payload=client, node_index=dict(), db=dict()) 

sets = [
    ( "TorsionDriveDataset", "OpenFF Gen 2 Torsion Set 2 Coverage" ),
    ( "TorsionDriveDataset", "OpenFF Gen 2 Torsion Set 2 Coverage 2" ),
    ( "OptimizationDataset", "OpenFF Gen 2 Opt Set 2 Coverage" )
]

for s in sets:
    ds = client.get_collection( *s)

    # Skip searching for Hessians on TD since they are (never) there?
    drop = ["Hessian"] if s[0] == "TorsionDriveDataset" else []

    # only get 2 objects per DS, just to test things
    QCA.build_index( ds, drop=drop, start=0, limit=2)

    # List specs and status
    specs = ds.list_specifications()
    for i in range(len(ds.list_specifications())):
        print(ds.status(specs.iloc[i]._name))

#}}}

# (Optional) Build the index out to include more than just 2 records
#{{{

# The "last" child here is the Optimization dataset, so we are
# asking to build a full index of the optimization data
# skel=True signifies only metadata and lightweight info is retreived
# This will enable iteration, but the heavy data e.g. the geometries will
# not be available

if False:
    opts = QCA.root().children[-1]
    QCA.expand_qca_dataset_as_tree( opts, skel=True, drop=["Hessian"])


#}}}

# Cache molecules locally by downloading final molecules in batches for speed
#{{{

# This is a specialized iterator that will download all final molecules
# from each of the optimizations, in all of the datasets in the index.
# This means all torsion drive optimizations and regular optimizations
# will be collected.
QCA.cache_optimization_minimum_molecules()

#}}}

# Example of how to iterate over the index
#{{{

# Iterate through the molecules, and combine data for the same molecule
# By default this is done by checking the 
# canonical_isomeric_explicit_hydrogen_mapped_smiles string in the entries.
# Creates a "folder" node, whose children are the entries which all
# have the same smiles, both torsiondrives and optimizations
for n in QCA.combine_by_entry():
    print(n.name, QCA.db[QCA[n.children[0]].payload]['entry'].attributes['canonical_smiles'])
    for e in n.children:
        record = QCA.db[QCA[e].payload]['data'] 
        if QCA[e].name == "TorsionDrive":
            print("    ", QCA[e].name, QCA[e].payload, record['status'][:], record['keywords'].dihedrals )
        else:
            print("    ", QCA[e].name, QCA[e].payload, record['status'][:])

#}}}

# Iterate entries, showing nodes to root
#{{{

for n in QCA.combine_by_entry():
    for e in n.children:
        e = QCA[e]
        for i,k in enumerate(QCA.node_iter_to_root(e)):
            print("  "*i,k)

#}}}

# Iterate molecules, showing nodes to root
#{{{
for n in QCA.node_iter_optimization_minimum(QCA.root()):
    for i,k in enumerate(QCA.node_iter_to_root(n)):
        print("  "*i,k)
#}}}

# Custom general iteration, where we iterate over molecules collated by:
# Molecule identity
#    Dihedral scanned
#        Final molecule ID and energy
for folder in QCA.combine_by_entry():
    print(folder)
    for entry in QCA.node_iter_entry(folder):
        print("    ", entry)
        for constraint in QCA.node_iter_depth_first(entry, select="Constraint"):
            print("      ", constraint)
            for molecule in QCA.node_iter_optimization_minimum(constraint):
                print("        ", molecule)

                # to give an idea, this will convert to a full QCArchive type
                qc_mol = qcelemental.models.molecule.Molecule.from_data(
                    QCA.db[molecule.payload]["data"])

# Utility functions to save data locally and run analysis, such as OpenMM energy
#{{{ 

# Some helper functions to run OpenMM energy calculations
# The default here is to calculate single points for each of the
# downloaded molecules, from QCA.cache_optimization_minimum_molecules
def run_openMM( QCA, ff, ffname, targets=None):
    from offsb.op import openmm
    oMM = openmm.OpenMMEnergy( ff, QCA, ffname)
    oMM.apply( targets=targets)
    save( oMM)
    return oMM

def save( tree):
    name = os.path.join(".", tree.name + ".p")
    print("Saving: ", tree.ID, "as", name, end=" ... ")
    tree.to_pickle( db=True, name=name)
    print( "{:12.1f} MB".format(os.path.getsize( name)/1024**2))

#}}}

# Save the index and db to disk
#{{{ 
# Save the index and db. This can be saved and loaded to disk without issue.
# Useful once all molecules have been downloaded, and all future processing
# can happen locally without having to reconnect to the remote server
save(QCA)
#}}}

# Example of how to load the data from disk
# Default is to save using the "name' as the filename (see constructors)
#{{{
if False:
    with open("QCA.p", 'rb') as fid:
        QCA = pickle.load(fid)
    with open("oMM.oFF-Parsley-uncons.p", 'rb') as fid:
        oMM = pickle.load(fid)
#}}}

# Example of how to get a list of all the entries
#{{{
ds_nodes = [ QCA[index] for index in QCA.root().children]
entries = list(QCA.iter_entry( ds_nodes))
#}}}

# Run the openMM calculation.
#{{{
# Optionally takes a targets arg, which will iterate over a select number
# of entries. Default is to iterate over all entries (in all datasets) in the
# index.
run_openMM( QCA, 
    'openff_unconstrained-1.0.0.offxml', 
    'oMM.oFF-Parsley-uncons')

