import numpy as np

from ..tools import const
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions


def qcmol_to_xyz(
    qcmol, atom_map=None, fnm=None, fd=None, comment="", precision="16.12f"
):
    syms = qcmol.symbols
    xyz = qcmol.geometry

    if not hasattr(xyz[0], "__iter__"):
        xyz = np.reshape(xyz, (-1, 3))

    if atom_map:
        syms = [syms[atom_map[i]] for i in atom_map]
        xyz = [xyz[atom_map[i]] for i in atom_map]

    precision = str(precision)

    xyzformat = " ".join(["{:4s}"] + list(["{:" + precision + "}"] * 3)) + '\n'

    header = [str(len(syms)) + "\n", str(comment) + "\n"]
    xyzstr = []
    for s, x in zip(syms, xyz):
        xyzline = xyzformat.format(s, *(x * const.bohr2angstrom))
        xyzstr.append(xyzline)
    if fnm:
        with open(fnm, "w") as myfd:
            myfd.write("".join(header))
            [myfd.write(line) for line in xyzstr]
    if fd:
        # print("HEADER", header)
        fd.write("".join(header))
        [fd.write(line) for line in xyzstr]
    return header + xyzstr
