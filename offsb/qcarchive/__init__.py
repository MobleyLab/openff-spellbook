from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from ..tools import const

from .qcatree import QCATree

def qcmol_to_xyz( qcmol, atom_map=None, fnm=None, fd=None, comment=""):
    syms = qcmol["symbols"]
    xyz  = qcmol["geometry"]
    
    if atom_map:
        idx = sorted( list(atom_map.keys()))
        syms = [ syms[ atom_map[ i]-1] for i in idx]
        xyz  = [  xyz[ atom_map[ i]-1] for i in idx]

    xyzformat = "{:4s} {:10.6f} {:10.6f} {:10.6}\n"

    header = [ str(len(syms)) + "\n", str(comment) + "\n"]
    xyzstr = []
    for s,x in zip( syms, xyz):
        xyzline = xyzformat.format( s, *( x * const.bohr2angstrom))
        xyzstr.append( xyzline)
    if fnm:
        with open( fnm, "w") as myfd:
            myfd.write( "".join( header))
            [ myfd.write( line) for line in xyzstr ]
    if fd:
        #print("HEADER", header)
        fd.write( "".join( header))
        [ fd.write( line) for line in xyzstr ]
    return header + xyzstr
