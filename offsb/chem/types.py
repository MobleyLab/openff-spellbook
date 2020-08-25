#/usr/bin/env python3

import itertools as it
import abc
import re

class ChemTypeComparisonException(Exception):
    pass

class ChemType(abc.ABC):
    """
    Represents a collection of bitfields which define a type in chemical space
    """

    def __init__(self, inv=False, endian=True):
        """
        A base type for Atom and Bond primitives which use bitfields to
        perform set logic on underlying fields. Fields represent variables
        in chemical space, such as symbol or bond order. These are discrete
        and usually stem from discrete forms of chemical space, e.g. SMILES.
        Parameters
        ----------
        inv: bool
            Whether the fields are inverted. This is required to ensure that
            leading bits are always 0.
        endian: bool
            True for big-endian, false for little-endian
        """

        self.inv = inv

        self._endian = endian

        for field in self._fields:
            setattr(self, field, self._trim(getattr(self,field)))

    def set_endianness(self, endian:bool ) -> None:
        """
        Set the ordering of the fields.
        Parameters
        ----------
        endian: bool
            True corresponds to setting to big-endian, False to little-endian.
        """

        if self._endian != endian:
            for field in self._fields:
                setattr(self, field, getattr(self, field)[::-1])
            self._endian = int(not self._endian)

    def is_null(self) -> bool:
        """
        Determines if the underlying type can represent a primitive.
        Specifically, is_null returns True if each field has at least one
        bit turned on.

        Returns: 
        --------
        ret: bool
            Whether the type can represent a primitive
        """

        if self.inv:
            return False

        present = "1"
        for name in self._fields:
            field = getattr(self, name)
            if present not in field:
                return True

        return False

    def copy(self):
        """
        Return a copy of self.
        Returns
        -------
        cls: type(cls)
           The new ChemType object
        """

        cls = self.__class__()

        for field in self._fields:
            setattr(cls, field, getattr(self, field))

        cls.inv = self.inv
        cls._endian = self._endian
        return cls

    @classmethod
    def from_string(cls, string):
        """
        """
        return cls.parse_string(string)

    @classmethod
    def _unpack_dat(cls, val:int, endian=1):
        dat = str("0"*(val+1))
        # val = str(val)
        if endian:
            val = "1" + dat[1:]
        else:
            val = dat[:-1] + "1"
        return val

    def _flip(self, bit:str, inv: bool) -> str:
        ret = bit
        if inv:
            ret = "1" if bit == "0" else "0"
        return ret

    def _explicit_flip(self):
        for field in self._fields:
            setattr(self, field, self._str_flip(getattr(self, field)))


    def _reduce(self):
        order = 1 if self._endian == 1 else -1
        sum = str(int(self.inv))
        for field in self._fields:
            sum += getattr(self, field)[::order]
        return sum

    def _reduce_longest(self, o):
        aorder = 1 if self._endian == 1 else -1
        border = 1 if o._endian == 1 else -1
        suma = str(int(self.inv))
        sumb = str(int(o.inv))
        for field in self._fields:
            a = getattr(self, field)
            b = getattr(o, field)
            pairs = list(it.zip_longest(a[::aorder], b[::border], fillvalue="0"))
            suma += "".join([self._flip(x[0], self.inv) for x in pairs])
            sumb += "".join([self._flip(x[1], o.inv) for x in pairs])
        return suma, sumb

    def _str_flip(self, a) -> str:
        return "".join([str(int(self._flip(i, True))) for i in a])

    def _str_and(self, a, b, inv=False, endian=1) -> str:
        aorder = -1 if self._endian == 1 else 1
        border = -1 if endian == 1 else 1
        pairs = list(it.zip_longest(a[::aorder], b[::border], fillvalue="0"))
        return "".join([str(int(
            self._flip(i, self.inv)+self._flip(j,inv) == "11"))
            for i,j in pairs[::aorder]])

    def _str_or(self, a, b, inv=False, endian=1) -> str:
        aorder = -1 if self._endian == 1 else 1
        border = -1 if endian == 1 else 1
        pairs = list(it.zip_longest( a[::aorder], b[::border], fillvalue="0"))
        return "".join([str(int(
            self._flip(i,self.inv)+self._flip(j,inv) != "00"))
            for i,j in pairs[::aorder]])

    def _str_xor(self, a, b, inv=False, endian=1) -> str:
        aorder = -1 if self._endian == 1 else 1
        border = -1 if endian == 1 else 1
        pairs = list(it.zip_longest(a[::aorder], b[::border], fillvalue="0"))
        return "".join([str(int(
            self._flip(i, self.inv)!=self._flip(j, inv) and 
            self._flip(i, self.inv)+self._flip(j, inv) != "00"))
            for i,j in pairs[::aorder]])

    def _check_sane_compare(self, o):
        if type(self) != type(o):
            raise ChemTypeComparisonException(
                "ChemType operations must use same type"
            )

    def _trim(self, dat):
        "Remove leading zeros"
        if self._endian == 1:
            if "1" in dat:
                return dat.lstrip("0")
            else:
                return "0"
        else:
            if "1" in dat:
                return dat.rstrip("0")
            else:
                return "0"

    def _dispatch_op(self, fn):

        args = [getattr(self, field) for field in self._fields]
        ret = fn(*args, inv=self.inv)
        ret = [self._trim(r) for r in ret]
        return self.from_data_string(*ret)

    def __and__(self, o):
        # bitwise and (intersection)

        self._check_sane_compare(o)

        a = self._dispatch_op(o.bitwise_and)

        # trailing bits will be 1 if both are inverted
        # so flip and set inverse st trailing bits are 0
        if self.inv and o.inv:
            a._explicit_flip()
            a.inv = True
        return a

    def __or__(self, o):
        # bitwise or (union)

        self._check_sane_compare(o)

        ret = self._dispatch_op(o.bitwise_or)

        # trailing bits will be 1 if one is inverted
        # so flip and set inverse st trailing bits are 0
        if self.inv or o.inv:
            ret._explicit_flip()
            ret.inv = True
        return ret

    def __xor__(self, o):
        # bitwise xor 

        self._check_sane_compare(o)

        ret = self._dispatch_op(o.bitwise_xor)

        # trailing bits will be 1 if only one is inverted
        # so flip and set inverse st trailing bits are 0
        if self.inv ^ o.inv:
            ret._explicit_flip()
            ret.inv = True

        return ret

    def __invert__(self):
        # negation (not a)

        a = self.copy()
        a.inv = not self.inv
        return a

    def __add__(self, o):
        # a + b is union

        return self.__or__(o) 

    def __sub__(self, o):
        # a - b is a marginal, note that a - b != b - a

        a = self.__xor__(o)
        return self & a

    def __contains__(self, o):
        self._check_sane_compare(o)
        return (self & o) == o

    def __eq__(self, o):
        a, b = self._reduce_longest(o)
        return a == b
        # return self._reduce() == o._reduce()

    def __lt__(self, o):
        a, b = self._reduce_longest(o)
        return a < b
        # return self._reduce() < o._reduce()

    def __gt__(self, o):
        a, b = self._reduce_longest(o)
        return a > b
        # return self._reduce() > o._reduce()

    def __le__(self, o):
        a, b = self._reduce_longest(o)
        return a <= b
        # return self._reduce() <= o._reduce()

    def __ge__(self, o):
        a, b = self._reduce_longest(o)
        return a >= b
        # return self._reduce() >= o._reduce()

    def __ne__(self, o):
        a, b = self._reduce_longest(o)
        return a != b

class AtomType(ChemType):
    """
    """

    def __init__(self, symbol:str = "0", X:str = "0", x:str = "0", H:str = "0",
        r:str = "0", aA:str = "0", inv:bool = False, endian:int = 1,
        ):
        """
        """

        self._fields = ["_symbol", "_X", "_x", "_H", "_r", "_aA"]
        # these are string bitfields, "" means Null
        self._symbol = symbol
        self._X = X
        self._x = x
        self._H = H
        self._r = r
        self._aA = aA

        super().__init__(inv=inv, endian=endian)


    @classmethod
    def from_data(cls, symbol:int, X:int, x:int, H:int, r:int, aA:int,
        inv:bool=False, endian=1):
        """
        """

        symbol = cls._unpack_dat(symbol, endian=endian)
        H = cls._unpack_dat(H, endian=endian)
        X = cls._unpack_dat(X, endian=endian)
        x = cls._unpack_dat(x, endian=endian)
        r = cls._unpack_dat(r, endian=endian)

        if endian:
            aA = "10" if aA == 1 else "01"
        else:
            aA = "10" if aA == 0 else "01"

        return cls(symbol, X, x, H, r, aA, inv, endian)


    @classmethod
    def from_data_string(cls, symbol:str = "0", X:str = "0", x:str = "0", H:str = "0",
        r:str = "0", aA:str = "0", inv:bool = False, endian:int = 1,
        ):
        """
        """
        return cls(symbol, X, x, H, r, aA, inv, endian)

    @classmethod
    def parse_string(self, string):
        """
        """

        # splits a single atom record, assumes a single primitive for now
        # therefore things like wildcards or ORs not supported

        # strip the [] brackets
        atom = string[1:-1]

        # The symbol number
        pat = re.compile(r"#([1-9][0-9]*)")
        ret = pat.search(atom)
        symbol = 0
        if ret:
            symbol = int(ret.group(1))

        # The number of bonds
        pat = re.compile(r"X([0-9][1-9]*)")
        ret = pat.search(atom)
        X = 0
        if ret:
            X = int(ret.group(1))

        # The number of bonds which are aromatic
        pat = re.compile(r"x([0-9][1-9]*)")
        ret = pat.search(atom)
        x = 0
        if ret:
            x = int(ret.group(1))

        # The number of bonds which are to hydrogen
        pat = re.compile(r"H([0-9][1-9]*)")
        ret = pat.search(atom)
        H = 0
        if ret:
            H = int(ret.group(1))

        # The size of the ring membership
        pat = re.compile(r"(!r|r[0-9]+)")
        ret = pat.search(atom)
        r = 0
        if ret:
            if ret.group(1) == "!r":
                r = 0
            else:
                # order is 0:0 1:None 2:None 3:1 4:2 2:r3 r4
                # since r1 and r2 are useless, r3 maps to second bit
                r = int(ret.group(1)[1:])-2

        # Whether the atom is considered aromatic
        pat = re.compile(r"([aA])$")
        ret = pat.search(atom)
        aA = 0
        if ret:
            aA = 0 if ret.group(1) == "A" else 1

        return self.from_data(symbol, X, x, H, r, aA)

    def __repr__(self):
        return "Syms: {} H: {} X: {} x: {} r: {} aA: {} Invert: {}".format(
            self._symbol, self._H, self._X, self._x, self._r, self._aA, self.inv)

    def bitwise_and(self, symbol, X, x, H, r, aA, inv=False) -> tuple:
        """
        """
        return self._bitwise_dispatch(symbol, X, x, H, r, aA,
            self._str_and, inv=inv)

    def bitwise_or(self, symbol, X, x, H, r, aA, inv=False) -> tuple:
        """
        """
        return self._bitwise_dispatch(symbol, X, x, H, r, aA,
            self._str_or, inv=inv)

    def bitwise_xor(self, symbol, X, x, H, r, aA, inv=False) -> tuple:
        """
        """
        return self._bitwise_dispatch(symbol, X, x, H, r, aA,
            self._str_xor, inv=inv)

    def bitwise_inv(self) -> tuple:
        """
        """
        symbol = self._symbol
        X = self._X
        x = self._x
        H = self._H
        r = self._r
        aA = self._aA
        inv = not self._inv 
        return symbol, X, x, H, r, aA, inv

    def _bitwise_dispatch(self, symbol, X, x, H, r, aA, fn, inv=False):
        symbol = fn(self._symbol, symbol, inv)
        X = fn(self._X, X, inv)
        x = fn(self._x, x, inv)
        H = fn(self._H, H, inv)
        r = fn(self._r, r, inv)
        aA = fn(self._aA, aA, inv)
        return symbol, X, x, H, r, aA

class BondType(ChemType):
    """
    """
    def __init__(self, order:str = "0", aA:str = "0", inv:bool = False,
            endian:int = 1):
        """
        """

        self._fields = ["_order", "_aA"]
        self._order = order
        self._aA = aA
        super().__init__(inv=inv, endian=endian)

    @classmethod
    def from_data(cls, order:int, aA:int, inv:bool=False, endian=1):
        """
        """

        order = cls._unpack_dat(order, endian=endian)

        if endian:
            aA = "10" if aA == 1 else "01"
        else:
            aA = "10" if aA == 0 else "01"

        return cls(order, aA, inv, endian)

    @classmethod
    def from_data_string(cls, order:str = "0", aA:str = "0", inv:bool = False,
            endian:int = 1):
        return cls(order, aA, inv, endian)

    @classmethod
    def parse_string(self, string):
        """
        splits a single atom record, assumes a single primitive for now
        therefore things like wildcards or ORs not supported
        """

        pat = re.compile(r"(.);(!?@)")
        ret = pat.search(string)
        
        order = 1

        if ret:
            sym = ret.group(1)
            if sym == "-":
                order = 1
            elif sym == "=":
                order = 2
            elif sym == "#":
                order = 3
            elif sym == ":":
                order = 4
            aA = 0 if ret.group(2) == "!@" else 1

        return self.from_data(order, aA)

    def __repr__(self):
        return "Order: {} aA: {} Invert: {}".format(
            self._order, self._aA, self.inv)

    def bitwise_dispatch(self, order, aA, fn, inv=False):
        order = fn(self._order, order, inv)
        aA = fn(self._aA, aA, inv)
        return order, aA

    def bitwise_and(self, order, aA, inv=False) -> tuple:
        return self.bitwise_dispatch(order, aA, self._str_and, inv=inv)

    def bitwise_or(self, order, aA, inv=False) -> tuple:
        return self.bitwise_dispatch(order, aA, self._str_or, inv=inv)

    def bitwise_xor(self, order, aA, inv=False) -> tuple:
        return self.bitwise_dispatch(order, aA, self._str_xor, inv=inv)

    def bitwise_inv(self) -> tuple:
        order = self._order
        aA = self._aA
        inv = not self._inv 
        return order, aA, inv

class ChemGraph(ChemType, abc.ABC):
    def __init__(self):
        pass

    def is_null(self):

        for field in self._fields:
            a = getattr(self, field)
            if a.is_null():
                return True

        return False

    @classmethod
    def from_string(cls, string):
        return cls._split_string(string)

    @classmethod
    def _smirks_splitter(cls, smirks, atoms):
        import re
        atom = r'(\[[^[.]*:?[0-9]*\])'
        bond = r'([^[.]*)'
        smirks = smirks.strip('()')
        if atoms <= 0:
            return tuple()
        pat_str = atom
        for i in range(1,atoms):
            pat_str += bond+atom
        pat = re.compile(pat_str)
        ret = pat.match(smirks)
        return ret.groups()

    def _check_sane_compare(self, o):
        for field in self._fields:
            a = getattr(self, field)
            b = getattr(o, field)
            a._check_sane_compare(b)

    def __and__(self, o):
        # bitwise and (intersection)
        self._check_sane_compare(o)
        ret_fields = []
        for field in self._fields:
            a = getattr(self, field)
            b = getattr(o, field)
            ret_fields.append(a & b)
            # if self.inv or o.inv:
            #     self.inv = True

        return self.from_data(*ret_fields)

    def __or__(self, o):
        # bitwise or (intersection)
        self._check_sane_compare(o)
        ret_fields = []
        for field in self._fields:
            a = getattr(self, field)
            b = getattr(o, field)
            ret_fields.append(a | b)
            # if self.inv or o.inv:
            #     self.inv = True

        return self.from_data(*ret_fields)

    def __xor__(self, o):
        # bitwise xor 
        self._check_sane_compare(o)
        ret_fields = []
        for field in self._fields:
            a = getattr(self, field)
            b = getattr(o, field)
            ret_fields.append(a ^ b)
            # if self.inv or o.inv:
            #     self.inv = True

        return self.from_data(*ret_fields)

    def __invert__(self):
        # negation (not a)
        ret_fields = []
        for field in self._fields:
            a = getattr(self, field)
            a.copy()
            a.inv = not self.inv
            ret_fields.append(a)

        return self.from_data(*ret_fields)

    def __add__(self, o):
        # a + b is union
        return self.__or__(o) 

    def __sub__(self, o):
        # a - b is a marginal, note that a - b != b - a
        a = self.__xor__(o)
        return self & a

    def __contains__(self, o):
        self._check_sane_compare(o)
        return (self & o) == o

    def __eq__(self, o):
        a, b = self._reduce_longest(o)
        return a == b
        # return self._reduce() == o._reduce()

    def __lt__(self, o):
        a, b = self._reduce_longest(o)
        return a < b
        # return self._reduce() < o._reduce()

    def __gt__(self, o):
        a, b = self._reduce_longest(o)
        return a > b
        # return self._reduce() > o._reduce()

    def __le__(self, o):
        a, b = self._reduce_longest(o)
        return a <= b
        # return self._reduce() <= o._reduce()

    def __ge__(self, o):
        a, b = self._reduce_longest(o)
        return a >= b
        # return self._reduce() >= o._reduce()

    def __ne__(self, o):
        a, b = self._reduce_longest(o)
        return a != b

class BondGraph(ChemGraph):
    def __init__(self, atom1=None, bond=None, atom2=None):
        if atom1 is None:
            atom1 = AtomType()
        if bond is None:
            bond = BondType()
        if atom2 is None:
            atom2 = AtomType()
        self._atom1 = atom1
        self._bond = bond
        self._atom2 = atom2
        self._fields = ["_atom1", "_bond", "_atom2"]


    @classmethod
    def from_data(cls, atom1, bond, atom2):
        return cls(atom1, bond, atom2)

    @classmethod
    def from_string_list(cls, string_list):
        atom1 = AtomType.from_string(string_list[0])
        bond1 = BondType.from_string(string_list[1])
        atom2 = AtomType.from_string(string_list[2])
        return cls(atom1, bond1, atom2)

    @classmethod
    def _split_string(cls, string):
        tokens = cls._smirks_splitter(string, atoms=2)
        return cls.from_string_list(tokens)
    def __repr__(self):
        return "(" + self._atom1.__repr__() + ") [" \
                + self._bond.__repr__() + "] (" \
                + self._atom2.__repr__() + ")"

class AngleGraph(ChemGraph):
    def __init__(self, atom1=None, bond1=None, atom2=None, bond2=None, atom3=None):
        if atom1 is None:
            atom1 = AtomType()
        if bond1 is None:
            bond1 = BondType()
        if atom2 is None:
            atom2 = AtomType()
        if bond2 is None:
            bond2 = BondType()
        if atom3 is None:
            atom3 = AtomType()
        self._atom1 = atom1
        self._bond1 = bond1
        self._atom2 = atom2
        self._bond2 = bond2
        self._atom3 = atom3
        self._fields = ["_atom1", "_bond1", "_atom2", "_bond2", "_atom3"]

    @classmethod
    def from_data(cls, atom1, bond1, atom2, bond2, atom3):
        return cls(atom1, bond1, atom2, bond2, atom3)

    @classmethod
    def from_string_list(cls, string_list):
        atom1 = AtomType.from_string(string_list[0])
        bond1 = BondType.from_string(string_list[1])
        atom2 = AtomType.from_string(string_list[2])
        bond2 = BondType.from_string(string_list[3])
        atom2 = AtomType.from_string(string_list[4])
        return cls(atom1, bond1, atom2, bond2, atom3)

    @classmethod
    def _split_string(cls, string):
        tokens = cls._smirks_splitter(string, atoms=3)
        return cls.from_string_list(tokens)

    def __repr__(self):
        return "(" + self._atom1.__repr__() + ") [" \
                + self._bond1.__repr__() + "] (" \
                + self._atom2.__repr__() + ") [" \
                + self._bond2.__repr__() + "] (" \
                + self._atom3.__repr__() + ")"

class DihedralGraph(ChemGraph):
    def __init__(self, atom1=None, bond1=None, atom2=None, bond2=None, atom3=None, bond3=None, atom4=None):
        if atom1 is None:
            atom1 = AtomType()
        if bond1 is None:
            bond1 = BondType()
        if atom2 is None:
            atom2 = AtomType()
        if bond2 is None:
            bond2 = BondType()
        if atom3 is None:
            atom3 = AtomType()
        if bond3 is None:
            bond3 = BondType()
        if atom4 is None:
            atom4 = AtomType()
        self._atom1 = atom1
        self._bond1 = bond1
        self._atom2 = atom2
        self._bond2 = bond2
        self._atom3 = atom3
        self._bond3 = bond3
        self._atom4 = atom4
        self._fields = ["_atom1", "_bond1", "_atom2", "_bond2", "_atom3",
                "_bond3", "_atom4"]

    @classmethod
    def from_data(cls, atom1, bond1, atom2, bond2, atom3, bond3, atom4):
        return cls(atom1, bond1, atom2, bond2, atom3, bond3, atom4)

    @classmethod
    def from_string_list(cls, string_list):
        atom1 = AtomType.from_string(string_list[0])
        bond1 = BondType.from_string(string_list[1])
        atom2 = AtomType.from_string(string_list[2])
        bond2 = BondType.from_string(string_list[3])
        atom3 = AtomType.from_string(string_list[4])
        bond3 = BondType.from_string(string_list[5])
        atom4 = AtomType.from_string(string_list[6])
        return cls(atom1, bond1, atom2, bond2, atom3, bond3, atom4)

    @classmethod
    def _split_string(cls, string):
        tokens = cls._smirks_splitter(string, atoms=4)
        return cls.from_string_list(tokens)

    def __repr__(self):
        return "(" + self._atom1.__repr__() + ") [" \
                + self._bond1.__repr__() + "] (" \
                + self._atom2.__repr__() + ") [" \
                + self._bond2.__repr__() + "] (" \
                + self._atom3.__repr__() + ") [" \
                + self._bond3.__repr__() + "] (" \
                + self._atom4.__repr__() + ")"

class OutOfPlaneGraph(DihedralGraph):
    def __init__(self, atom1, bond1, atom2, bond2, atom3, bond3, atom4):
        super().__init__(atom1, bond1, atom2, bond2, atom3, bond3, atom4)

    def __repr__(self):
        return "(" + self._atom1.__repr__() + ") [" \
                + self._bond1.__repr__() + "] (" \
                + self._atom2.__repr__() + ") [" \
                + self._bond2.__repr__() + "] ((" \
                + self._atom3.__repr__() + ")) [" \
                + self._bond3.__repr__() + "] (" \
                + self._atom4.__repr__() + ")"
    
