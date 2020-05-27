
from abc import ABCMeta, abstractmethod
from enum import Flag

NEW = -1
ON = 1
OFF = 0
CLEAN = 3
DIRTY = 2

# ON and CLEAN is 1 + 2 = 3
# ON and not CLEAN is 1 + 0 = 1
# OFF and CLEAN is 0 + 2 = 2
# OFF and not CLEAN is 0 + 0 = 0

STATE = { DIRTY: "Dirty", CLEAN: "Clean", NEW: "New", ON: "On", OFF: "Off"}


def stamp( obj):
    from datetime.datetime import now
    return now()

class Node():
    index = 0
    def __init__( self, parent=None, index=None, name="", tree=None, payload=None, stamp=None, state=(not 2**ON)):
        self.parent = parent
        if parent is not None:
            parent.add( self)
        self.children = []
        if index is None:
            self.index = str(Node.index)
            Node.index += 1
        else:
            self.index = str( index)

        self.name = name
        self.stamp = stamp
        self.payload = payload
        self.state = state # -1 is new, 0 is clean, 1 is dirty
        self.tree = tree

        IDcomponents = [self.index, self.name]
        if tree:
            IDcomponents.insert( 0, self.tree.ID)
        self.ID=".".join( IDcomponents)

    
    def skel( self):
        return Node( index=self.index, name=self.name, state=self.state)

    def copy( self):
        return Node( index=self.index, name=self.name, state=self.state, payload=self.payload)

    def set_state( self, state):
        self.state = state

    def __repr__( self):
        return "<Node Name={} Tree={} IDX={} Payload={}>".format(
                self.name, self.tree, self.index,  
                "No" if self.payload is None else str( self.payload))
        return "<Node Name={} Tree={} IDX={} State={} Payload={}>".format(
                self.name, self.tree, self.index, STATE.get( self.state), 
                "No" if self.payload is None else str( self.payload))

    def add( self, v):
        assert isinstance(v, Node)
        self.children.append( v.index)
        v.parent = self.index
