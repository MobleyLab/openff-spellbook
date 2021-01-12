import offsb.chem.types
from numpy import inf

def test_bits():

    v = offsb.chem.types.BitVec(maxbits=10)
    assert v.bits() == 0
    v[0] = True
    assert v.bits() == 1
    v[3] = True
    assert v.bits() == 2

    v = ~v
    assert v.bits() == inf
    assert v.bits(maxbits=True) == 8

    assert sum([1 for x in v]) == 8

def test_getitem():

    v = offsb.chem.types.BitVec(maxbits=10)
    v[:5] = True
    breakpoint()
    ans = v[2:8:2]
    assert len(ans) == 3
    assert sum(ans) == 2

    assert len(v[0:2]) == 2

    v.clear()
    v[:] = True
    assert len(v[0:2]) == 2
    

test_getitem()
