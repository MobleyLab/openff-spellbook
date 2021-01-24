import offsb.chem.types
# These ->
# False (Syms:  1000010 H:      1 X:  10010 x:      1 r:      1 aA:      1) [Order:     10 aA:      1] (Syms:  1000000 H:     10 X:  10000 x:      1 r:      1 aA:      1) [Order:     10 aA:      1] (Syms:  100000000 H:     10 X:    100 x:      1 r:      1 aA:      1)
# False (Syms:  1000010 H:     11 X:  10010 x:      1 r:      1 aA:      1) [Order:     10 aA:      1] (Syms:  1000000 H:     10 X:  10000 x:      1 r:      1 aA:      1) [Order:     10 aA:      1] (Syms:  100000000 H:     10 X:    100 x:      1 r:      1 aA:      1)
# False (Syms:     10 H:      1 X:     10 x:      1 r:      1 aA:      1) [Order:     10 aA:      1] (Syms:  1000000 H:     10 X:  10000 x:      1 r:      1 aA:      1) [Order:     10 aA:      1] (Syms:  100000000 H:     10 X:    100 x:      1 r:      1 aA:      1)
# True (Syms:  1000010 H:   1001 X:  10010 x:      1 r:      1 aA:      1) [Order:     10 aA:      1] (Syms:  100000000 H:     10 X:    100 x:      1 r:      1 aA:      1) [Order:     10 aA:      1] (Syms:  1000000 H:     10 X:  10000 x:      1 r:      1 aA:      1)

# should all be in here
# ipdb> p self.db[lbl]["data"]["group"]
# (Syms:  101000010 H:  11111 X:  11111 x:      1 r:      1 aA:      1) [Order:     10 aA:      1] (Syms:  100000010 H:  11111 X:  11111 x:      1 r:      1 aA:      1) [Order:     10 aA:      1] (Syms:  101000010 H:  11111 X:  11111 x:      1 r:      1 aA:      1)

def test_torsion_group_contains():
    a = offsb.chem.types.TorsionGroup()
    b = offsb.chem.types.TorsionGroup()
    assert a in b
    a._atom2._symbol[1] = True
    b._atom3._symbol[1] = True
    assert a in b

def test_torsion_group_marginal():

    a = offsb.chem.types.TorsionGroup()
    b = offsb.chem.types.TorsionGroup()
    ret = a == b
    assert ret

    ret = a != b
    assert not ret

    a._atom2._symbol[1] = True
    b._atom3._symbol[1] = True
    ret = a != b
    assert ret

    ret = a == b
    assert not ret

def test_torsion_graph():

    s = "[*:1]-[#6X4:2]-[#6X4:3]-[*:4]"
    a = offsb.chem.types.TorsionGraph.from_string(s)
    s = "[*:1]-[#6X4:2]-[*:3]-[*:4]"
    b = offsb.chem.types.TorsionGraph.from_string(s)
    assert a in b
    assert b not in a

def test_angle_graph():

    s = "[*:1]-[#6X4:2]-[#1:3]"
    a = offsb.chem.types.AngleGraph.from_string(s)
    s = "[#1:1]-[#6X4:2]-[*:3]"
    b = offsb.chem.types.AngleGraph.from_string(s)
    assert a == b
    assert a in b
    assert b in a

def test_angle_graph_smarts():

    s = "[#1:1]-[#6X4:2]-[*:3]"
    a = offsb.chem.types.AngleGraph.from_string(s)

    # print(s)
    # print("Before")
    # print(a)
    # print(a.to_smarts())
    a._atom3._H[2:4] = False
    a._atom3._X[2:4] = False
    print("After")
    print(a)
    print(a.to_smarts())

def test_check_bond_contains():

    a = offsb.chem.types.BondType()
    a._order[1] = True
    a._aA[0] = True

    b = offsb.chem.types.BondType()
    b._order[1:] = True
    b._aA[0] = True

    print("a", a)
    print("b", b)
    print("b in a", b in a)
    print("a in b", a in b)

    assert a in b
    assert b not in a

def test_bond_contains():
    s = "[#6X3:1]-[#8H1X2:2]"
    a = offsb.chem.types.BondGraph.from_string(s)
    s = "[#6X3:1]-[#8X1:2]"
    b = offsb.chem.types.BondGraph.from_string(s)
    assert a not in b
    assert b not in a

def test_is_valid():
    s = "[*:1]~[#6X4:2]-[*:3]"
    a = offsb.chem.types.AngleGraph.from_string(s)
    assert a.is_valid()

    s = "[#6X3:1]-[#16X2:2]-[#1:3]"
    a = offsb.chem.types.AngleGraph.from_string(s)
    assert a.is_valid()

    s = "[#6X4;r3:1]-;@[#6X4;r3:2]-[#6X3;r6:3]:[#7X2;r6:4]"
    a = offsb.chem.types.TorsionGraph.from_string(s)
    if not a.is_valid():
        breakpoint()
    assert a.is_valid()

    s = "[*:1]-[#8X2:2]@[#6X3:3]~[*:4]"
    a = offsb.chem.types.TorsionGraph.from_string(s)
    if not a.is_valid():
        breakpoint()
    assert a.is_valid()

    s = "[*:1]-[#7X4,#7X3,#7X2-1:2]-[*:3]"
    a = offsb.chem.types.AngleGraph.from_string(s)
    if not a.is_valid():
        breakpoint()
    assert a.is_valid()

    s = "[#6X3,#7:1]~;@[#8;r:2]~;@[#6X3,#7:3]"
    a = offsb.chem.types.AngleGraph.from_string(s)
    if not a.is_valid():
        breakpoint()
    assert a.is_valid()

def test_angle_bit_flip():
    # turning off this bit caused the ordering to flip, leading
    # to the bit appled to the RHS (instead of the LHS)
    # turned off some reordering stuff; so this is a good test for
    # ensuring proper function of bit application
    s = "[*:1]-[#8:2]-[!H0;!x0:3]"
    a = offsb.chem.types.AngleGraph.from_string(s)
    b = offsb.chem.types.AngleGraph()
    b._atom1._r[4] = 1
    smarts = (a - b).to_smarts()
    ans = smarts == "[!r6:1]-[#8:2]-[!H0;!x0:3]"
    assert ans

    s = "[!x0:1]-[#8:2]-[*:3]"
    a = offsb.chem.types.AngleGraph.from_string(s)
    b = offsb.chem.types.AngleGraph()
    b._atom3._H[1] = 1
    smarts = (a - b).to_smarts()
    ans = smarts == "[!x0:1]-[#8:2]-[!H1:3]"
    assert ans
