
import offsb.chem.types
from openforcefield.topology.molecule  import Molecule
import rdkit.Chem

def test_negated_atom_smarts():

    a = offsb.chem.types.AtomType.from_string("[*]")

    a._H[0:2] = False
    a._r[1:4] = False
    a._symbol[0:2] = False
    a._aA[1] = False
    a._X[:2] = False
    a._x[:] = False
    a._x[0] = True
    smarts = a.to_smarts(tag=True)

    mol = Molecule.from_smiles("CCO")

    top = mol.to_topology()
    matches = top.chemical_environment_matches(smarts)
    assert len(matches) == 2

    assert smarts == '[!#1;!H1!H0;!X1;!r5!r4!r3;A:1]'

def test_negated_bond():

    breakpoint()
    a = offsb.chem.types.BondType.from_string("~")

    a._order[0:4] = False
    a._aA[0] = False
    smarts = a.to_smarts()

    print(smarts)


    # assert len(matches) == 2

    # assert smarts == '[!#1;!H1!H0;!X1;!r5!r4!r3;A:1]'

test_negated_bond()
