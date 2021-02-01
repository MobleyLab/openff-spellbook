
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

    for i in ["!#1", "!H1", "!H0", "!X1", "x0", "!r5", "!r4", "!r3", "A", ":1"]:
        # assert smarts == '[!#1;!H1!H0;!X1;x0;!r5!r4!r3;A:1]'
        # guard against future cases that might reorder and give false negatives
        assert i in smarts

def test_negated_bond():

    a = offsb.chem.types.BondType.from_string("~")

    a._order[0:4] = False
    a._aA[0] = False
    smarts = a.to_smarts()

    print(smarts)


    # assert len(matches) == 2

    # assert smarts == '[!#1;!H1!H0;!X1;!r5!r4!r3;A:1]'

def test_smarts():

    s = "[#1;X1:1]-[#6;H3;X4:2]"
    a = offsb.chem.types.BondGraph.from_string(s, sorted=False)
    s = "[#6;H2;X4:1]-[#1;X1:2]"
    b = offsb.chem.types.BondGraph.from_string(s, sorted=False)

    c = offsb.chem.types.BondGraph()
    c._atom1._symbol[1] = True

    print("subtract", c)
    print("a", a)
    print(a - c, "good?", a & c == c, "contains?", c in a)
    print("b", b)
    print(b - c, "good?", b & c == c, "contains?", c in b)


test_smarts()
