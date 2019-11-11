
from .node import *
import collections
import numpy as np
from abc import (ABC, abstractmethod)
import os
import copy
import warnings
import pprint

def link_iter_depth_first( t):
    for c in t.link.values():
        yield from link_iter_depth_first( c)
    yield t

def link_iter_breadth_first( t):
    for c in t.link.values():
        yield c
    for c in t.link.values():
        yield from link_iter_breadth_first( c)

def link_iter_dive( t):
    yield t
    for c in t.link.values():
        yield from link_iter_dive( c)

def link_iter_to_root( t):
    yield t
    if t.source is not None:
        yield from link_iter_to_root( t.source, select, state)

class Tree( ABC):
    index=0
    """ Basically just a structre that holds an index for fast lookups.
        Also holds associations for links. Linking will create a new tree with
        empty payloads; will keep IDs


    """
    def __init__( self, name, root_payload=None, node_index=None, db=None, index=None, payload=None):

        if db is not None:
            self.db = db
        else:
            self.db = dict()
        print("Tree database using type", type( self.db))

        if root_payload is not None:
            ROOT = "ROOT"
            root = Node( name=ROOT )
            root.payload = ROOT

            self.db.__setitem__( ROOT, { "data": root_payload } )


            root.index = '0'
            self.root_index = root.index
            self.node_index = { root.index : root }
        elif node_index is not None:
            self.node_index = node_index
        else:
            self.node_index = dict()


        self.link = {}
        self.name = name
        self.modified = set()
        self.N = 1
        if index is None:
            self.index = 'TREE-' + str( Tree.index)
            Tree.index += 1
        else:
            self.index = str( index)
        #self.node_index = nodes
        #for node in self.node_index:
        #    self.node_index.get( node).tree = self.name
        self.n_levels = 0 if len(self.node_index) == 0 else 1 #max( [node_level( node) for node in nodes.values()])
        self.n_nodes = len( self.node_index)
        #self.root = None if self.n_levels == 0 else get_root( list(nodes.values())[0])
        self.ID = ".".join([self.index, self.name])

    def to_pickle(self, db=True, index=True, name=None):
        import pickle
        if name is None:
            name = self.name + ".p"
        #self.isolate()
        if not db:
            tmp = self.db
            self.db = None
        if not index:
            tmp_index = self.node_index
            self.node_index = None
        with open( name, 'wb') as fid:
            pickle.dump( self, fid)
        if not db:
            self.db = tmp
        if not index:
            self.node_index = tmp_index

    @abstractmethod
    def to_pickle_str( self):
        pass

    def set_root(self, root):
        self.node_index = {node.index : node for node in self.node_iter_depth_first( root)}
        self.n_levels = max( [node_level( node) for node in self.node_index.values()])
        self.n_nodes = node_descendents( root) + 1
        self.root = root
        self.root.tree = self.name

    def register_modified( self, node, state=DIRTY):
        self.modified.add( node.index)
        node.state = state

    def link_tree( self, tree):
        self.link[tree.index] = tree
        tree.source = self

    def link_generate( self):
        tree = self.copy_skel()
        self.link_tree( tree)
        tree.source = self
        return tree

    def join( self, nodes, fn=link_iter_breadth_first):
        ret = {}
        if not hasattr(nodes, "__iter__"):
            nodes = [nodes]
        for node in nodes:
            ret[ node.index] = {}
            for tree in fn( self):
                ret[ node.index][ tree.name] = tree.node_index.get( node.index) 
        return ret

    def copy_skel( self):
        """ provides a new tree with no payload or links """
        nodes = {node.index: node for node in [node.skel() for node in self.node_index.values()]}
        for node in nodes.values():
            [node.add( nodes.get( v.index)) for v in self.node_index.get( node.index).children]
        tree = Tree( nodes=nodes, name=self.name)
        [tree.register_modified( node) for node in self.node_index.values()]
        return tree

    def register_db_payload( self, key, payload):
        self.db.__setitem__(key, payload)

    def add(self, parent_index, child):
        """
        takes a parent node index and a fresh constructed node
        inserts the payload into the db and creates a reference in the node
        """

        assert self.N not in self.node_index
        assert isinstance( parent_index, str)

        parent = self.node_index.get( parent_index)
        child.index = str(self.N) #+ '-' + child.index
        self.N += 1
        parent.add( child)
        child.tree=self.name
        self.node_index.__setitem__( child.index, child)
        
        #self.obj_index.update( child.payload)
        #self.node_index[child.] = child
        #self.n_levels = max(self.n_levels, 1 + node_level( parent))
        #self.n_nodes += 1
        #self.register_modified( child, state=NEW)

        #for tree in self.link.values():
        #    tree.add( parent_index, child.skel())

    def assemble( self):
        for ID in self.node_index:
            node = self.node_index.get( ID)
            if isinstance( node.parent, str):
                node.parent = self.node_index.get( node.parent)
            for i,_ in enumerate( node.children):
                if isinstance( node.children[i], str):
                    node.children[i] = self.node_index.get( node.children[i])
        if isinstance( self.root, str):
            self.root = self.node_index.get( self.root)

    def yield_if( self, v, select, state):
        if select is None or select == v.name :
            if state is None or state == v.state :
                yield v

    def node_iter_depth_first( self, v, select=None, state=None):
        for c in v.children:
            c = self.node_index.get( c)
            yield from self.node_iter_depth_first( c, select, state)
        yield from self.yield_if( v, select, state)

    def node_iter_breadth_first(self, v, select=None, state=None):
        if v.parent is None:
            yield v
        for c in v.children:
            c = self.node_index.get( c)
            yield from self.yield_if( c, select, state)
        for c in v.children:
            c = self.node_index.get( c)
            yield from self.node_iter_breadth_first( c, select, state)

    def node_iter_dive( self, v, select=None, state=None):
        yield from self.yield_if( v, select, state)
        for c in v.children:
            c = self.node_index.get( c)
            yield from self.node_iter_dive( c, select, state)

    def node_iter_to_root( self, v, select=None, state=None):
        yield from self.yield_if( v, select, state)
        if v.parent is not None:
            parent = self.node_index.get( v.parent)
            yield from self.node_iter_to_root( parent, select, state)

    def get_root( self, node):
        if node.parent is None:
            return node
        return self.get_root( self.node_index.get( node.parent))

    def node_depth( self, node):
        n = node
        l = 0
        while n.parent is not None:
            n = self.node_index.get( n.parent)
            l += 1
        return l

    def node_descendents( self, node):
        if len( node.children) == 0:
            return 0
        return 1 + sum([ self.node_descendents( self.node_index.get( v)) for v in node.children])

    def node_level( self, node):
        l = 1
        v = node
        while v.parent is not None:
            v = self.node_index.get( v.parent)
            l += 1
        return l 

class PartitionTree( Tree):
    """ A parition tree holds indices and applies them to nodes 
        Importantly, partition trees are not trees... 
        They are a dictionary, where it copies the source tree index as keys,
        and puts data in there

        so if the index has node.index -> '4', and QCATree sets node.payload -> 2234521
        QCA -> { 2234521 : { meta : obj, data : obj }}

        then partition has { '2234521' : { meta: obj, data : obj }} 

        so this means we have 3 objects:
            the index
            the QCA data
            the paritition data

        and are separate.
    """
    
    def __init__( self, source_tree, name):
        self.source = source_tree

        print( "Building PartitionTree", name)
        #nodes = {node.index: node for node in [node.skel() for node in source_tree.node_index.values()]}
        #self.node_index = source_tree.node_index
        #for node in nodes.values():
        #    print( node, node.children)
        #    for v in source_tree.node_index.get( node.index).children:
        #        #print("Connecting ", node)
        #        #print("            to ", nodes.get( v.index))
        #        v = nodes.get( v.index)
        #        print( v)

        #        node.add( v)
        super().__init__( node_index=source_tree.node_index, name=name)
        [self.register_modified( node) for node in self.node_index.values()]
        #source_tree.link_tree(self)
    def to_pickle( self, name=None, index=False, db=True):
        import pickle
        if not index:
            tmp = self.node_index
            self.node_index = None
        tmp_source = self.source
        self.source = tmp_source.ID
        if not db:
            tmp_db = self.db
            self.db = None
        super().to_pickle( name=name, db=db)
        self.source = tmp_source
        if not index:
            self.node_index = True
        if not db:
            self.db = tmp_db

    def to_pickle_str( self):
        import pickle
        self.isolate()
        return pickle.dumps( self)

#    def to_pickle( self, db=False, name=None):
#        import pickle
#        if name is None:
#            name = self.name + ".p"
#        #self.isolate()
#        with open( name, 'wb') as fid:
#            pickle.dump( self, fid)

    def isolate( self):
        pass
        #for ID in self.node_index:
        #    node = self.node_index.get( ID)
        #    if node.parent is not None:
        #        if not isinstance(node.parent, str):
        #            node.parent = node.parent.index
        #    for i,_ in enumerate( node.children):
        #        if not isinstance(node.children[i], str):
        #            node.children[i] = node.children[i].index
        #if not isinstance( self.root, str):
        #    self.root = self.root.index
        #if not isinstance( self.source, str):
        #    self.source = self.source.name

    def associate( self, source):
        self.assemble()
        source.link_tree( self)
        self.root = self.node_index.get( self.root.index)

    def apply( self):
        pass


class EnergyTree( PartitionTree):
    """
    Apply some energy functions such as from e.g. openMM
    """
    def __init__( self):
        pass


class TreeOperation( PartitionTree):

    def __init__( self, source_tree, name):
        super().__init__( source_tree, name)
        #for k in self.source.index:
        #    self.index[ k] = self.source.index[ k].skel()

    @abstractmethod
    def op(self, node, partition):
        pass
    
    def apply( self, targets=None):
        calcs = 0
        self.source.apply( targets=targets)
        #print( self.source.db.get( 'QCP-1762049'))
        if targets is None:
            entries = list(self.source.source.iter_entry())
        else:
            entries = targets
        #print( "Op will iterate on", entries )
        if not hasattr( entries, "__iter__"):
            entries = [entries]
        #print( "Op will iterate on", len(entries), "entries")
        for entry in entries:
            mol_calcs = 0
            #self.source.apply( entry)
            obj = self.source.db.get( self.source.node_index.get( entry.index).payload )
            #print( "QUERY FROM SOURCE DB", self.source.node_index.get( entry.index))

            #print( "ENTRY", self.node_index.get( entry.index), obj, end="\n")

            masks = obj.get( "data")
            #print( "Have ", len(masks), "masks and", len( list(node_iter_depth_first( entry, select="Molecule"))), "structures" )
            #print( masks)
            #for masks in self.source.node_index[ entry.index].payload.values():
                #print( "  MASK", mask)
            gen = list(self.node_iter_depth_first( entry, select="Molecule"))
            #print( "Op will iterate on", len(gen), "molecules")
            for mol_node in gen:
                mol = self.source.source.db.get( mol_node.payload)
                for mask in masks:
                    #print(entry.name, mol, mask)
                    ret = {}
                    if mol_node.payload in self.db:
                        ret = self.db.get( mol_node.payload)
                    else:
                        self.db.__setitem__( mol_node.payload, {})
                    #entry_this_mol = next( node_iter_to_root( mol, select="TorsionDrive"))
                    
                    #print( "Molecule from", entry_this_mol.payload.get( "meta").name, "has shape", mol.payload.get( "geometry").shape)
                    ret[ tuple( mask)] = self.op( mol.get( "data"), [i-1 for i in mask])
                    self.db[ mol_node.payload].update( ret)
                    #self.node_index[ mol.index].state = CLEAN
                    mol_calcs += 1
                    #print( "    MOL", mol, self.node_index[ mol.index].payload.keys() )
            calcs += mol_calcs
            #print( ": calcs = ", mol_calcs)
        print(self.name + " calculated: {}".format( calcs))
        #self.partition.update()
        #for k in self.index:
        #    if self.index[ k].payload is None:
        #        self.index[ k].payload = {}

