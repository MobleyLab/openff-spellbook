
import offsb.chem.types

def test_angle_bit_flip():
    # turning off this bit caused the ordering to flip, leading
    # to the bit appled to the RHS (instead of the LHS)
    # turned off some reordering stuff; so this is a good test for
    # ensuring proper function of bit application
    s = "[#1:1]-[#8:2]-[#6:3]"
    a = offsb.chem.types.AngleGraph.from_string(s)
    s = "[#6:1]-[#8:2]-[#1:3]"
    b = offsb.chem.types.AngleGraph.from_string(s)

    # These are technically the same in terms of chemistry
    ans = a == b
    assert ans

    # The string will be flipped
    ans = a.to_smarts() == b.to_smarts()
    assert not ans

    a.align_to(b)

    ans = a.to_smarts() == b.to_smarts()
    assert ans

