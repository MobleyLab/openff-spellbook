
import pickle
import sys
import os
import numpy as np
from . import qcatree as qca
from ..op import geometry
from ..op import openforcefield
from ..search import smiles
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
        ds = client.get_collection("torsiondrivedataset", "openff group1 torsions")
        QCA.build_index( ds, drop=["Optimization"])
        #ds = client.get_collection("optimizationdataset", "openff optimization set 1")
        #QCA.build_index( ds, drop=["Hessian"])
        QCA.to_pickle( db=True)
    #QCA.set_root( client_node)

    if 1:
        #print( QCA.db.keys())
        #client = QCA.db.get( QCA.node_index.get( QCA.root_index).payload).get( "data")
        dsopt = client.get_collection("optimizationdataset", "openff optimization set 1")
        QCA.build_index( dsopt, drop=["Gradient"])
    #roche_opt = client.get_collection("optimizationdataset", "openff optimization set 1")
    #roche_opt_node = node.Node(payload=roche_opt, name=roche_opt.data.name, index=roche_opt.data.id)
    #QCA.add(client_node.index, roche_opt_node)
    #QCA.expand_qca_dataset_as_tree(QCA.root.children[-1], skel=True)
    if 0:
        #QCA.cache_torsiondriverecord_minimum_molecules()
        QCA.cache_optimization_minimum_molecules()
    if 1:
        QCA.to_pickle( db=False)
        QCA.to_pickle( name=QCA.name + ".db.p", index=False, db=True)

    return QCA


def process( QCA=None):

    if QCA is None:
        with open("QCA.p", 'rb') as fid:
            QCA = pickle.load(fid)
    if QCA.db is None:
        with open("QCA.db.p", 'rb') as fid:
            QCA.db = pickle.load(fid)
    #QCA.cache_optimization_minimum_molecules( QCA.root)

    def save( tree):
        name = os.path.join(".", tree.name + ".p")
        print("Saving: ", tree.ID, "as", name, end=" ... ")
        tree.to_pickle( db=True, name=name)
        print( "{:12.1f} MB".format(os.path.getsize( name)/1024**2))
        #with open( name, 'wb') as fid:
        #    fid.write( obj)
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

    def run_oFF09( QCA):
        oFF09 = openforcefield.OpenForceFieldTree('smirnoff99Frosst-1.0.9.offxml', QCA, 'oFF-1.0.9')
        oFF09.apply( targets=entries)
        save( oFF09)

    def run_oFF10( QCA):
        oFF10 = openforcefield.OpenForceFieldTree('smirnoff99Frosst-1.1.0.offxml', QCA, 'oFF-1.1.0')
        oFF10.apply( targets=entries)
        save( oFF10)

    def run_oMM09( QCA):
        from ..op import openmm
        oMM10 = openmm.OpenMMEnergy('smirnoff99Frosst-1.0.9.offxml', QCA, 'oMM.oFF-1.0.9')
        oMM10.apply( targets=entries)
        save( oMM10)
    def run_oMM10( QCA):
        from ..op import openmm
        oMM10 = openmm.OpenMMEnergy('smirnoff99Frosst-1.1.0.offxml', QCA, 'oMM.oFF-1.1.0')
        oMM10.apply( targets=entries)
        save( oMM10)

    #run_bonds(      QCA)
    #run_angles(     QCA)
    #run_torsions(   QCA)
    #run_outofplane( QCA)
    #run_oFF09(      QCA)
    #run_oFF10(      QCA)
    run_oMM09(       QCA)
    run_oMM10(       QCA)

