#/usr/bin/env python3

import itertools as it
import abc
import re
import numpy as np

class ChemTypeComparisonException(Exception):
    pass

class BitVec:
    
    def __init__(self, vals=None, inv=False):

        self.__slots__ = ["_v", "_inv"]
        if vals is None:
            self._v = np.array([False], dtype=bool)
        else:
            self._v = np.array(vals, dtype=bool)
        self._inv = inv
        self.trim()

    def __getitem__(self, i):
        if i >= self._v.shape[0]:
            return self.inv
        else:
            return self._v[i]

    def __setitem__(self, i, v:bool):

        if isinstance(i, slice):

            if i.stop is None and i.start is None and i.step is None:
                # if x[:] = y is used, clear and maybe invert
                self.clear()
                if v:
                    # x[:] = True so set to zeros and invert
                    self.inv = True
                return

            l = i.stop
            start = 0 if i.start is None else i.start
            end = max(self._v.shape[0] if i.stop is None else i.stop, i.start)
            step = 1 if i.step is None else i.step

            if end >= self._v.shape[0]:
                if self.inv != v:
                    diff = end - self._v.shape[0] + 1
                    self._v = np.concatenate((self._v, [False]*diff))

            # Want to set the tailing bits to True, so flip start
            # and set inv
            if i.stop is None and self.inv ^ v:
                self._v[0:start][:] = ~self._v[0:start]
                # self._v[start:][:] = self.inv ^ v
                self._inv = not self._inv
            for j in range(start, end, step):
                self._v[j] = v

        elif isinstance(i, int):
            if i >= self._v.shape[0]:
                if self.inv != v:
                    diff = i - self._v.shape[0] + 1
                    self._v = np.concatenate((self._v, [False]*diff))
                    self._v[i] =  self.inv ^ v
            else:
                self._v[i] = self.inv ^ v
        else:
            raise Exception("Using this datatype for setitem not supported")
        self.trim()

    def __len__(self):
        return self._v.shape[0]

    def __repr__(self):
        neg="~" if self._inv else " "
        outstr = "".join([str(int(v)) for v in self._v[::-1]]) 
        return neg+outstr

    def trim(self):

        if self._v.shape[0] < 2:
            return

        drop=0
        for i in range(1,self._v.shape[0]):
            if not self._v[-i]:
                drop+=1
            else:
                break
        if drop > 0:
            self._v = self._v[:-drop]

    def explicit_flip(self):
        np.logical_not(self._v, out=self._v)

    def is_null(self):
        if self._inv:
            return False
        else:
            return not np.any(self._v)

    def clear(self):
        self._v[:] = False
        self.trim()
        self.inv = False

    def copy(self):
        return BitVec(self._v, self._inv)

    def reduce_longest(self, o):
        """
        """
        
        pairs  = list(it.zip_longest(self.v, o.v, fillvalue=False))
        pairs  = np.vstack(([(self.inv,o.inv)],pairs))

        suma, sumb = np.packbits(pairs,axis=0)[0]
        return suma, sumb

    @property
    def inv(self):
        return self._inv
    
    @property
    def v(self):
        return self._v

    @inv.setter
    def inv(self, switch:bool):
        self._inv = switch

    def _logical_op(self, o, fn):
        # this is the section with the same length
        l = min(self.v.shape[0], o.v.shape[0])

        # negate bits if inverted
        a = ~self.v[:l] if self.inv else self.v[:l]
        b = ~o.v[:l] if o.inv else o.v[:l]

        # perform the logical operation
        a = fn(a,b)

        # now do the remainder, can shortcut this since the short end
        # is just comparing its inv flag

        same_length = self.v.shape[0] == o.v.shape[0]
        if not same_length:
            if l == self.v.shape[0]:
                # o is longer

                # invert if needed, then compare with others invert
                c = ~o.v[l:] if o.inv else o.v[l:]
                c = fn(c,self.inv)
            else:
                # self is longer
                c = ~self.v[l:] if self.inv else self.v[l:]
                c = fn(c,o.inv)

            a = np.concatenate((a,c))

        inv = False
        if fn(self.inv, o.inv):
            a = ~a
            inv = True

        return BitVec(a, inv)

    def __and__(self, o):
        return self._logical_op(o, np.logical_and)

    def __or__(self, o):
        return self._logical_op(o, np.logical_or)

    def __xor__(self, o):
        return self._logical_op(o, np.logical_xor)

    def __invert__(self):
        return BitVec(self._v, inv=not self._inv)

    def __add__(self, o):
        return self | o

    def __sub__(self, o):
        return self & ( self ^ o )


class ChemType(abc.ABC):
    """
    Represents a collection of bitfields which define a type in chemical space
    """

    def __init__(self, inv=False):
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
        """

        self.inv = inv

        for field in self._fields:
            setattr(self, field, getattr(self,field))

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
            vec:BitVec = getattr(self, name)
            if vec.is_null():
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
            setattr(cls, field, getattr(self, field).copy())

        cls.inv = self.inv
        return cls

    @classmethod
    def from_string(cls, string):
        """
        """
        return cls.parse_string(string)

    def _flip(self, bit:str, inv: bool) -> str:
        ret = bit
        if inv:
            ret = "1" if bit == "0" else "0"
        return ret

    def _explicit_flip(self):
        for field in self._fields:
            arr:BitVec = getattr(self, field)
            arr.explicit_flip()


    def _reduce(self):
        sum = 0
        for field in self._fields:
            vec:BitVec = getattr(self, field)
            sum += vec.reduce()
        return sum

    def reduce_longest(self, o):
        
        suma = 0
        sumb = 0
        for field in self._fields:
            
            a:BitVec = getattr(self, field)
            b:BitVec = getattr(o, field)
            x,y = a.reduce_longest(b)
            suma += x
            sumb += y

        return suma, sumb


    def _check_sane_compare(self, o):
        if type(self) != type(o):
            raise ChemTypeComparisonException(
                "ChemType operations must use same type"
            )

    def _trim(self, dat):
        "Remove leading zeros"

        for field in self._fields:
            vec:BitVec = getattr(self, field)
            vec.trim()

    def _dispatch_op(self, fn):

        args = [getattr(self, field) for field in self._fields]
        ret = fn(*args, inv=self.inv)
        ret = [self._trim(r) for r in ret]
        return self.from_data_string(*ret)

    def __and__(self, o):
        # bitwise and (intersection)

        self._check_sane_compare(o)

        ret = []
        for field in self._fields:
            a_vec:BitVec = getattr(self, field)
            b_vec:BitVec = getattr(o, field)
            ret.append(a_vec & b_vec)

        return self.from_data(*ret)

    def __or__(self, o):
        # bitwise or (union)

        self._check_sane_compare(o)

        ret = []
        for field in self._fields:
            a_vec:BitVec = getattr(self, field)
            b_vec:BitVec = getattr(o, field)
            ret.append(a_vec | b_vec)

        return self.from_data(*ret)

    def __xor__(self, o):
        # bitwise xor 

        self._check_sane_compare(o)

        ret = []
        for field in self._fields:
            a_vec:BitVec = getattr(self, field)
            b_vec:BitVec = getattr(o, field)
            ret.append(a_vec ^ b_vec)

        return self.from_data(*ret)

    def __invert__(self):
        # negation (not a)

        ret = []
        for field in a._fields:
            a_vec:BitVec = getattr(self, field)
            a_vec = ~a_vec.copy()
            ret.append(a_vec)

        return self.from_data(*ret)

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
        a, b = self.reduce_longest(o)
        return a == b

    def __lt__(self, o):
        a, b = self.reduce_longest(o)
        return a < b

    def __gt__(self, o):
        a, b = self.reduce_longest(o)
        return a > b

    def __le__(self, o):
        a, b = self.reduce_longest(o)
        return a <= b

    def __ge__(self, o):
        a, b = self.reduce_longest(o)
        return a >= b

    def __ne__(self, o):
        a, b = self.reduce_longest(o)
        return a != b

class AtomType(ChemType):
    """
    """

    def __init__(self, symbol:BitVec=None, X:BitVec=None, x:BitVec=None,
            H:BitVec=None, r:BitVec=None, aA:BitVec=None, inv:bool=False
        ):
        """
        """

        self._fields = ["_symbol", "_X", "_x", "_H", "_r", "_aA"]
        # these are string bitfields, "" means Null
        if symbol is None:
            symbol = BitVec()
        self._symbol = symbol

        if X is None:
            X = BitVec()
        self._X = X

        if x is None:
            x = BitVec()
        self._x = x

        if H is None:
            H = BitVec()
        self._H = H

        if r is None:
            r = BitVec()
        self._r = r

        if aA is None:
            aA = BitVec()
        self._aA = aA

        super().__init__(inv=inv)


    @classmethod
    def from_data(cls, symbol:BitVec, X:BitVec, x:BitVec, H:BitVec, r:BitVec,
            aA:BitVec,
        inv:bool=False):
        """
        """

        cls = cls()
        cls._symbol = symbol
        cls._X = X
        cls._x = x
        cls._H = H
        cls._r = r
        cls._aA = aA
        return cls

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
        symbol = BitVec()
        if ret:
            symbol[int(ret.group(1))] = True

        # The number of bonds
        pat = re.compile(r"X([0-9][1-9]*)")
        ret = pat.search(atom)
        X = BitVec()
        if ret:
            X[int(ret.group(1))] = True

        # The number of bonds which are aromatic
        pat = re.compile(r"x([0-9][1-9]*)")
        ret = pat.search(atom)
        x = BitVec()
        if ret:
            x[int(ret.group(1))] = True

        # The number of bonds which are to hydrogen
        pat = re.compile(r"H([0-9][1-9]*)")
        ret = pat.search(atom)
        H = BitVec()
        if ret:
            H[int(ret.group(1))] = True

        # The size of the ring membership
        pat = re.compile(r"(!r|r[0-9]+)")
        ret = pat.search(atom)
        r = BitVec()
        if ret:
            if ret.group(1) == "!r":
                r[0] = True
            else:
                # order is 0:0 1:None 2:None 3:1 4:2 2:r3 r4
                # since r1 and r2 are useless, r3 maps to second bit
                r[int(ret.group(1)[1:])-2] = True

        # Whether the atom is considered aromatic
        pat = re.compile(r"([aA])$")
        ret = pat.search(atom)
        aA = BitVec()
        if ret:
            if ret.group(1) == "a":
                aA[1] = True
            else:
                aA[0] = True

        return self.from_data(symbol, X, x, H, r, aA)

    def __repr__(self):
        return "Syms: {} H: {} X: {} x: {} r: {} aA: {} Invert: {}".format(
            self._symbol, self._H, self._X, self._x, self._r, self._aA, self.inv)

class BondType(ChemType):
    """
    """
    def __init__(self, order:BitVec = None, aA:BitVec = None, inv:bool = False):
        """
        """

        self._fields = ["_order", "_aA"]
        if order is None:
            order = BitVec()
        if aA is None:
            aA = BitVec()
        self._order = order
        self._aA = aA
        super().__init__(inv=inv)

    @classmethod
    def from_data(cls, order:BitVec, aA:BitVec, inv:bool=False):
        """
        """
        return cls(order, aA, inv)

    @classmethod
    def parse_string(self, string):
        """
        splits a single atom record, assumes a single primitive for now
        therefore things like wildcards or ORs not supported
        """

        pat = re.compile(r"(.);(!?@)")
        ret = pat.search(string)
        
        order = BitVec()
        aA = BitVec()

        if ret:
            sym = ret.group(1)
            if sym == "-":
                order[1] = True
            elif sym == "=":
                order[2] = True
            elif sym == "#":
                order[3] = True
            elif sym == ":":
                order[4] = True

            aA[1] = True if ret.group(2) == "!@" else False
            aA[0] = True if ret.group(2) == "@"  else False

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
        super().__init__()

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
            a = a.copy()
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
        a, b = self.reduce_longest(o)
        return a == b

    def __lt__(self, o):
        a, b = self.reduce_longest(o)
        return a < b

    def __gt__(self, o):
        a, b = self.reduce_longest(o)
        return a > b

    def __le__(self, o):
        a, b = self.reduce_longest(o)
        return a <= b

    def __ge__(self, o):
        a, b = self.reduce_longest(o)
        return a >= b

    def __ne__(self, o):
        a, b = self.reduce_longest(o)
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
        super().__init__()

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
    def __init__(self, atom1=None, bond1=None, atom2=None, bond2=None,
            atom3=None):
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
    def __init__(self, atom1=None, bond1=None, atom2=None, bond2=None,
            atom3=None, bond3=None, atom4=None):
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
    
