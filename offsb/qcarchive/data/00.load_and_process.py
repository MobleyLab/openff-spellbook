#!/usr/bin/env python3
import pickle
import sys
import os
import numpy as np

import offsb.qcarchive.qcatree as qca
from offsb.op import geometry
from offsb.op import openforcefield
from offsb.op import openmm
from offsb.search import smiles
import treedi.node as Node




def load():
    
    import qcfractal.interface as ptl

    NAME = "QCA"
    # this builds the index, starting with the client node
    NAMEP = NAME + ".p"
    if os.path.exists( NAMEP ):
        with open( NAMEP, 'rb') as fid:
            QCA = pickle.load( fid)
        if QCA.db is None:
            with open( NAME + ".db.p", 'rb') as fid:
                QCA.db = pickle.load( fid).db

    else:

        client = ptl.FractalClient()
        client_node = Node.Node( payload=client, name="client")
        QCA = qca.QCATree( NAME, root_payload=client, node_index=None, db=None )

        DS_NAME = "OpenFF Trivalent Nitrogen Set 1"
        DS_NAME = "SMIRNOFF Coverage Torsion Set 1"

        DS_TYPE = "TorsionDriveDataset"
        DS_NAME = "OpenFF Gen 2 Torsion Set 2 Coverage"
        client.get_collection( DS_TYPE, DS_NAME)
        ds = client.get_collection( DS_TYPE, DS_NAME)

        drop=[]
        # since we know there are no Hessians, skip looking for them
        #drop.append("Hessian")
        QCA.build_index( ds, drop=drop)

        DS_TYPE = "OptimizationDataset"
        DS_NAME = "OpenFF Gen 2 Opt Set 2 Coverage"
        ds = client.get_collection( DS_TYPE, DS_NAME)
        QCA.build_index( ds, drop=drop)

        # this will download the final structure of *all* minimizations found
        # for a gridopt, this will just be the final structure of each point
        QCA.cache_optimization_minimum_molecules()

        # save the index and data to disk for future analysis
        QCA.to_pickle( db=False)
        QCA.to_pickle( name=QCA.name + ".db.p", index=False, db=True)

    return QCA


def process( QCA=None):

    # load QCA from disk if not given
    if QCA is None:
        with open("QCA.p", 'rb') as fid:
            QCA = pickle.load(fid)
    if QCA.db is None:
        with open("QCA.db.p", 'rb') as fid:
            QCA.db = pickle.load(fid)

    def save( tree):
        name = os.path.join(".", tree.name + ".p")
        print("Saving: ", tree.ID, "as", name, end=" ... ")
        tree.to_pickle( db=True, name=name)
        print( "{:12.1f} MB".format(os.path.getsize( name)/1024**2))
        #with open( name, 'wb') as fid:
        #    fid.write( obj)

    # these will be the datasets processed
    # assume they are the children of the root (level 2)
    ds_nodes = [ QCA.node_index.get( index) for index in QCA.node_index.get( QCA.root_index).children]

    print( "ds_nodes", ds_nodes)
    entries = list(QCA.iter_entry( ds_nodes))

    def run_bonds( QCA):
        pairs = smiles.SmilesSearchTree( "*~*", QCA, name="pairs")
        bonds = geometry.BondOperation( pairs, name="bonds")
        bonds.apply( targets=entries)
        save( bonds)

    def run_angles( QCA):
        triples = smiles.SmilesSearchTree( "*~*~*", QCA, name="triples")
        angles = geometry.AngleOperation( triples, name="angles")
        angles.apply( targets=entries)
        save( angles)

    def run_torsions( QCA):
        linquads = smiles.SmilesSearchTree("*~*~*~*", QCA, name="linquads")
        torsions = geometry.TorsionOperation(linquads, name="torsions")
        torsions.apply( targets=entries)
        save( torsions)

    def run_outofplane( QCA):
        pyramidquads = smiles.SmilesSearchTree("*~*(~*)~*", QCA, name="pyramidquads")
        outofplane = geometry.ImproperTorsionOperation(pyramidquads, name="outofplane")
        outofplane.apply( targets=entries)
        save( outofplane)


    def run_oFFParsley( QCA):
        oFF10 = openforcefield.OpenForceFieldTree('openff-1.0.0.offxml', QCA, 'oFF-Parsley')
        oFF10.apply( targets=entries)
        save( oFF10)

    #################################
    # Calculate energy from OpenMM
    # Note that this will call/create an openforcefield.OpenForceFieldTree
    # automatically, i.e. it must label them before calculating the energy

    def run_openMM( QCA, ff, ffname, targets=None):
        oMM = openmm.OpenMMEnergy( ff, QCA, ffname)
        oMM.apply( targets=targets)
        save( oMM)
        return oMM

    #run_bonds(      QCA)
    #run_angles(     QCA)
    #run_torsions(   QCA)
    #run_outofplane( QCA)
    #run_oFF10(      QCA)
    #run_oFFParsley( QCA)
    #run_oMM10(      QCA)
    #run_openMM( QCA, 'smirnoff99Frosst-1.1.0.offxml',  'oMM.oFF-frosst1.1.0')
    #run_openMM( QCA, 'openff-1.0.0.offxml', 'oMM.oFF-Parsley-constr' ) 
    run_openMM( QCA, 'openff_unconstrained-1.0.0.offxml', 'oMM.oFF-Parsley-uncons',targets=[entries[0]]) 


if __name__ == "__main__":
    QCA = load()
    #process(QCA)
