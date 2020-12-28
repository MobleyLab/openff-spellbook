import offsb.chem.types
def test_bond_contains():
    a = offsb.chem.types.BondType.parse_string("@")
    assert a.is_valid()

def test_bond_contains():
    a = offsb.chem.types.BondType()
    a._order[4] = True
    b = offsb.chem.types.BondType()
    b._order[1] = True
    b._aA[:] = True
    assert a not in b

def test_bond():
    s = "[#6X3:1]-[#6!H1X3:2]"
    a = offsb.chem.types.BondGraph.from_string(s)
    b = offsb.chem.types.BondGraph()
    b._atom2._H[1] = True

    # this one is tricky since this will match the LHS
    # but if we want to take out this bit, we need to 
    assert b  in a

    assert (a - b) == a

    b._atom2._H[2] = True

    assert b  in a

    assert (a - b) != a


def test_atom():
    s = "[#6!H1X3:2]"
    a = offsb.chem.types.AtomType.from_string(s)
    b = offsb.chem.types.AtomType()
    b._H[1] = True

    assert b not in a

    s = "[#6X4;r3:1]"
    a = offsb.chem.types.AtomType.from_string(s)
