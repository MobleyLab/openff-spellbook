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

