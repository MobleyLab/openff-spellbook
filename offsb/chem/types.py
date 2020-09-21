# /usr/bin/env python3

import abc
import itertools as it
import re
from typing import List

import numpy as np


class ChemTypeComparisonException(Exception):
    pass


class AtomTypeInvalidException(Exception):
    pass


class AtomTypeInvalidException(Exception):
    pass


class BondTypeInvalidException(Exception):
    pass


class BitVec:

    __slots__ = ["_v", "_inv"]

    def __init__(self, vals=None, inv=False):

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

    def __setitem__(self, i, v: bool):

        if isinstance(i, slice):

            if i.stop is None and i.start is None and i.step is None:
                # if x[:] = y is used, clear and maybe invert
                self.clear()
                if v:
                    # x[:] = True so set to zeros and invert
                    self.inv = True
                return

            start = 0 if i.start is None else i.start
            end = max(self._v.shape[0] if i.stop is None else i.stop, start)
            step = 1 if i.step is None else i.step

            if end >= self._v.shape[0]:
                if self.inv != v:
                    diff = end - self._v.shape[0] + 1
                    self._v = np.concatenate((self._v, [False] * diff))

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
                    self._v = np.concatenate((self._v, [False] * diff))
                    self._v[i] = self.inv ^ v
            else:
                self._v[i] = self.inv ^ v
        else:
            raise Exception("Using this datatype for setitem not supported")
        self.trim()

    def __len__(self):
        return self._v.shape[0]

    def __repr__(self):
        neg = "~" if self._inv else " "
        outstr = "".join([str(int(v)) for v in self._v[::-1]])
        return neg + outstr

    def trim(self):

        if self._v.shape[0] < 2:
            return

        drop = 0
        for i in range(1, self._v.shape[0]):
            if not self._v[-i]:
                drop += 1
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

    def reduce(self):
        """"""
        ret = np.sum(np.packbits(self._v), dtype=np.int32)
        return ret

    def reduce_longest(self, o):
        """"""

        pairs = list(it.zip_longest(self.v, o.v, fillvalue=False))
        pairs = np.vstack(([(self.inv, o.inv)], pairs))

        suma, sumb = np.packbits(pairs, axis=0).sum(axis=0, dtype=np.int32)
        return suma, sumb

    @property
    def inv(self):
        return self._inv

    @property
    def v(self):
        return self._v

    @inv.setter
    def inv(self, switch: bool):
        self._inv = switch

    def _logical_op(self, o, fn):
        # this is the section with the same length
        l = min(self.v.shape[0], o.v.shape[0])

        # negate bits if inverted
        a = ~self.v[:l] if self.inv else self.v[:l]
        b = ~o.v[:l] if o.inv else o.v[:l]

        # perform the logical operation
        a = fn(a, b)

        # now do the remainder, can shortcut this since the short end
        # is just comparing its inv flag

        same_length = self.v.shape[0] == o.v.shape[0]
        if not same_length:
            if l == self.v.shape[0]:
                # o is longer

                # invert if needed, then compare with others invert
                c = ~o.v[l:] if o.inv else o.v[l:]
                c = fn(c, self.inv)
            else:
                # self is longer
                c = ~self.v[l:] if self.inv else self.v[l:]
                c = fn(c, o.inv)

            a = np.concatenate((a, c))

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
        return self & (self ^ o)


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
            setattr(self, field, getattr(self, field))

    def _recurse_fields(self, fields, pos=[]):

        if len(pos) == len(fields):
            ret = [field[i] for field, i in zip(fields, pos)]

            dat = {}
            assert len(self._field_vars) == len(ret)
            for k, v in zip(self._field_vars, pos):
                dat[k] = v

            if all(ret) and self.is_valid(self.from_ints(dat)):
                yield pos.copy()
            return
            # return [field[i] for field,i in zip(fields,pos)]
        l_pos = len(pos)
        pos.append(0)
        # print("field", l+1, "/", len(fields), "has length", len(fields[l]))
        for i in range(len(fields[l_pos])):
            pos[-1] = i
            # print("querying", i)
            yield from self._recurse_fields(fields, pos=pos.copy())

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

        for name in self._fields:
            vec: BitVec = getattr(self, name)
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
        """"""
        return cls.parse_string(string)

    def _flip(self, bit: str, inv: bool) -> str:
        ret = bit
        if inv:
            ret = "1" if bit == "0" else "0"
        return ret

    def _explicit_flip(self):
        for field in self._fields:
            arr: BitVec = getattr(self, field)
            arr.explicit_flip()

    def reduce(self):
        sum = 0
        for field in self._fields:
            vec: BitVec = getattr(self, field)
            sum += vec.reduce()
        return sum

    def reduce_longest(self, o):

        suma = 0
        sumb = 0
        for field in self._fields:

            a: BitVec = getattr(self, field)
            b: BitVec = getattr(o, field)
            x, y = a.reduce_longest(b)
            suma += x
            sumb += y

        return suma, sumb

    def _check_sane_compare(self, o):
        if type(self) != type(o):
            raise ChemTypeComparisonException("ChemType operations must use same type")

    def _trim(self, dat):
        "Remove leading zeros"

        for field in self._fields:
            vec: BitVec = getattr(self, field)
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
            a_vec: BitVec = getattr(self, field)
            b_vec: BitVec = getattr(o, field)
            ret.append(a_vec & b_vec)

        return self.from_data(*ret)

    def __or__(self, o):
        # bitwise or (union)

        self._check_sane_compare(o)

        ret = []
        for field in self._fields:
            a_vec: BitVec = getattr(self, field)
            b_vec: BitVec = getattr(o, field)
            ret.append(a_vec | b_vec)

        return self.from_data(*ret)

    def __xor__(self, o):
        # bitwise xor

        self._check_sane_compare(o)

        ret = []
        for field in self._fields:
            a_vec: BitVec = getattr(self, field)
            b_vec: BitVec = getattr(o, field)
            ret.append(a_vec ^ b_vec)

        return self.from_data(*ret)

    def __invert__(self):
        # negation (not a)

        ret = []
        for field in self._fields:
            a_vec: BitVec = getattr(self, field)
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
    """"""

    def __init__(
        self,
        symbol: BitVec = None,
        X: BitVec = None,
        x: BitVec = None,
        H: BitVec = None,
        r: BitVec = None,
        aA: BitVec = None,
        inv: bool = False,
    ):
        """"""

        self._field_vars = ["S", "H", "X", "x", "r", "aA"]
        self._fields = ["_symbol", "_H", "_X", "_x", "_r", "_aA"]
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
    def from_data(
        cls,
        symbol: BitVec,
        H: BitVec,
        X: BitVec,
        x: BitVec,
        r: BitVec,
        aA: BitVec,
        inv: bool = False,
    ):
        """"""

        cls = cls()
        cls._symbol = symbol
        cls._H = H
        cls._X = X
        cls._x = x
        cls._r = r
        cls._aA = aA
        return cls

    @classmethod
    def from_ints(cls, data, inv: bool = False):
        """"""

        cls = cls()
        vec = BitVec()
        vec[data["S"]] = True
        cls._symbol = vec.copy()

        vec = BitVec()
        vec[data["X"]] = True
        cls._X = vec.copy()

        vec = BitVec()
        vec[data["x"]] = True
        cls._x = vec.copy()

        vec = BitVec()
        vec[data["H"]] = True
        cls._H = vec.copy()

        vec = BitVec()
        vec[data["r"]] = True
        cls._r = vec.copy()

        vec = BitVec()
        vec[data["aA"]] = True
        cls._aA = vec.copy()

        return cls

    @classmethod
    def parse_string(self, string):
        """"""

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
                r[int(ret.group(1)[1:]) - 2] = True

        # Whether the atom is considered aromatic
        pat = re.compile(r"([aA])$")
        ret = pat.search(atom)
        aA = BitVec()
        if ret:
            if ret.group(1) == "a":
                aA[1] = True
            else:
                aA[0] = True

        return self.from_data(symbol, H, X, x, r, aA)

    def __repr__(self):
        return "Syms: {} H: {} X: {} x: {} r: {} aA: {}".format(
            self._symbol, self._H, self._X, self._x, self._r, self._aA
        )

    def to_primitives(self, bond_limit=4):
        """
        Converts the representation to an enumeration of all possible
        primitive types that are valid.
        """
        if self.is_null():
            raise AtomTypeInvalidException("Does not define any primitive")
        """
        my order is S, H, X, r, aA
        """

        terms = list()
        vals = [
            x
            for x in self._recurse_fields(
                [getattr(self, field) for field in self._fields], pos=[]
            )
        ]
        for line in vals:
            rstr = "!r" if line[-2] == 0 else "r{}".format(line[-2] + 2)
            aA = "A" if line[-1] == 0 else "a"
            term = "[#{}H{}X{}x{}{}{}]".format(*line[:-2], rstr, aA)

            terms.append(term)

        return terms

    @classmethod
    def is_valid(cls, atom) -> bool:

        sym = len(atom._symbol) - 1
        H = len(atom._H) - 1
        x = len(atom._x) - 1
        X = len(atom._X) - 1
        r = len(atom._r) - 1

        # if X0 or X1, then can't be in a ring
        if X < 3 and r > 1:
            return False

        # The sum of H and x must be <= X
        if X < x + H:
            return

        # Everything else is acceptable?
        return not atom.is_null()


def next_bond_def(atom: AtomType, fix=None) -> AtomType:
    """
    Generate a +1 step in bond def, that will produce a valid atom
    fix determines what cannot be changed in the operation
    """
    return atom.copy()


def prev_bond_def(atom: AtomType, fix=None) -> AtomType:
    """
    Generate a +1 step in bond def, that will produce a valid atom
    fix determines what cannot be changed in the operation
    """
    return atom.copy()

    return True


def iterate_atom_connections(
    fix: dict = {}, init: dict = {}, direction=1, order=["S", "X", "r", "x", "H"]
):
    """
    rules:
    x + H <= X
    if x == 0, r == 0
    if X < 2, x == 0

    # if x > 1, r == unbound # no constraint
    if X >= 2 x == unbound # no constraint

    """
    # X = fix.get("X", 2)

    # rmax = 8
    # Sallowed = [1,6,7,8]

    # rrange = [init.get("r", 0), rmax - init.get("r", 0)]
    # xrange = [init.get("x", 0), X]
    # Hrange = [init.get("H", 0), X]
    # Xrange = [init.get("X", 0), X]

    # Srange = [init.get("S", Sallowed[0]), Sallowed[-1]]
    # # Srange = [s for s in range(Srange[0], Srange[1] + 1) if s in Sallowed]

    # if direction == -1:
    #     rrange = rrange[::-1]
    #     xrange = xrange[::-1]
    #     Hrange = Hrange[::-1]
    #     Xrange = Xrange[::-1]
    #     Srange = Srange[::-1]

    # def next_S(S, X, r, x, H, direction=1, allowed=[1,6,7,8]):
    #     S = S + 1 if direction > 0 else S - 1
    #     if S > 0 and S in allowed:
    #         yield S

    # def next_X(S, X, r, x, H, direction=1, max_val=4):
    #     X = X + 1 if direction > 0 else X - 1
    #     if X > 0 and X <= max_val:
    #         yield X

    # def next_r(S, X, r, x, H, direction=1):
    #     r = r + 1 if direction > 0 else r - 1
    #     if x + H <= X and r > 0:
    #         yield r

    # def next_x(S, X, r, x, H, direction=1):
    #     x = x + 1 if direction > 0 else x - 1
    #     if not (x == 0 and r > 0) and x > 0:
    #         yield x

    # def next_H(S, X, r, x, H, direction=1):
    #     H = H + 1 if direction > 0 else r - 1
    #     if x + H <= X and H > 0:
    #         yield H

    # vars = []
    # table = []
    # for o in order:
    #     success = False
    #     if o == 'S':
    #         vars.append(Srange)
    #         table.append(next_S)
    #     elif o == 'X':
    #         vars.append(Xrange)
    #         table.append(next_X)
    #     elif o == 'r':
    #         vars.append(rrange)
    #         table.append(next_r)
    #     elif o == 'x':
    #         vars.append(Hrange)
    #         table.append(next_x)
    #     elif o == 'H':
    #         vars.append(Hrange)
    #         table.append(next_H)

    # terms = []

    # # vec = [next(fn(i)) for fn,x in zip(table, vars) for i in range(*x)]
    # vec = [next(i) for var in vars for i in range(*var)]
    # # make sure vec is valid

    # terms.append(vec)
    # for i,var in enumerate(vars):
    #     vec[i] = table[i](var)
    #     terms.append()
    # return terms

    X = fix.get("X", 2)

    rmax = 8
    Sallowed = [1, 6, 7, 8]

    rrange = [init.get("r", 0), rmax - init.get("r", 0)]
    xrange = [init.get("x", 0), X + 1]
    Hrange = [init.get("H", 0), X + 1]
    Xrange = [init.get("X", 0), X + 1]

    Srange = [init.get("S", Sallowed[0]), Sallowed[-1]]
    Srange = [s for s in range(Srange[0], Srange[1] + 1) if s in Sallowed]

    if direction == -1:
        rrange = rrange[::-1]
        xrange = xrange[::-1]
        Hrange = Hrange[::-1]
        Xrange = Xrange[::-1]
        Srange = Srange[::-1]

    terms = []
    for S in Srange:
        for H in range(*Hrange, direction):
            for x in range(*xrange, direction):
                for r in range(*rrange, direction):

                    if x + H > X:
                        continue
                    # if x == 0 and r > 0:
                    #     continue
                    if r == 1 or r == 2:
                        continue

                    rstr = "!r" if r == 0 else "r{}".format(r)
                    term = "[#{}H{}X{}x{}{}a]".format(S, H, X, x, rstr)
                    terms.append(term)
                    term = "[#{}H{}X{}x{}{}A]".format(S, H, X, x, rstr)
                    terms.append(term)
    return terms


class BondType(ChemType):
    """"""

    __slots__ = ["_order", "_aA"]

    def __init__(self, order: BitVec = None, aA: BitVec = None, inv: bool = False):
        """"""

        self._field_vars = ["Order", "aA"]
        self._fields = ["_order", "_aA"]
        if order is None:
            order = BitVec()
        if aA is None:
            aA = BitVec()
        self._order = order
        self._aA = aA
        super().__init__(inv=inv)

    @classmethod
    def from_data(cls, order: BitVec, aA: BitVec, inv: bool = False):
        """"""
        return cls(order, aA, inv)

    @classmethod
    def from_ints(cls, data, inv: bool = False):
        """"""

        cls = cls()
        vec = BitVec()
        vec[data["Order"]] = True
        cls._order = vec.copy()

        vec = BitVec()
        vec[data["aA"]] = True
        cls._aA = vec.copy()

        return cls

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

            aA[1] = True if ret.group(2) == "@" else False
            aA[0] = True if ret.group(2) == "!@" else False

        return self.from_data(order, aA)

    def __repr__(self):
        return "Order: {} aA: {}".format(self._order, self._aA)

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

    @classmethod
    def is_valid(cls, bond) -> bool:

        # Everything else is acceptable?
        return not bond.is_null()

    def to_primitives(self):
        """
        Converts the representation to an enumeration of all possible
        primitive types that are valid.
        """
        if self.is_null():
            raise BondTypeInvalidException("Does not define any bond primitive")
        """
        my order is order;aromatic
        """

        terms = list()
        vals = [
            x
            for x in self._recurse_fields(
                [getattr(self, field) for field in self._fields], pos=[]
            )
        ]
        for line in vals:
            aA = "!@" if line[-1] == 0 else "@"
            bond_lookup = {1: "-", 2: "=", 3: "#", 4: ":"}
            bond = bond_lookup[line[0]]
            term = "{};{}".format(bond, aA)
            terms.append(term)

        return terms


class ChemGraph(ChemType, abc.ABC):
    def __init__(self):
        super().__init__()

    def is_null(self):

        for field in self._fields:
            a = getattr(self, field)
            if a.is_null():
                return True

        return False

    def reduce(self):

        reduce = 0
        for field in self._fields:
            a = getattr(self, field)
            reduce += a.reduce()
        return reduce

    @classmethod
    def from_string(cls, string):
        return cls._split_string(string)

    @classmethod
    def _smirks_splitter(cls, smirks, atoms):
        import re

        atom = r"(\[[^[.]*:?[0-9]*\])"
        bond = r"([^[.]*)"
        smirks = smirks.strip("()")
        if atoms <= 0:
            return tuple()
        pat_str = atom
        for i in range(1, atoms):
            pat_str += bond + atom
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

    def to_primitives(self):
        prims = []
        for field in self._fields:
            obj = getattr(self, field)
            prims.append(set(obj.to_primitives()))

        ret = []
        visited = []
        tracker = BondGraph()
        i = 0
        for a1 in prims[0]:
            for bnd in prims[1]:
                for a2 in prims[2]:
                    prim = a1 + bnd + a2
                    #ret.append(prim)

                    # bt = BondGraph.from_string_list([a1, bnd, a2])
                    # bt = bt | BondGraph.from_string_list([a2, bnd, a1])

                    # new = (bt - tracker).reduce()
                    # if new > 0:
                    #     ret.append(prim)
                    #     ret.append(a2 + bnd + a1)
                    #     tracker += bt
                    # else:
                    #     i += 1
                    #     print(i, "No new info:", bt)


                    if  prim not in visited and a2 + bnd + a1 not in visited:
                    # if a1 >= a2:
                        ret.append(prim)
                        visited.append(prim)
                    else:
                        i += 1
                        print(i, "No new info:", prim)

        return ret

    def __repr__(self):
        return (
            "("
            + self._atom1.__repr__()
            + ") ["
            + self._bond.__repr__()
            + "] ("
            + self._atom2.__repr__()
            + ")"
        )


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
        atom3 = AtomType.from_string(string_list[4])
        return cls(atom1, bond1, atom2, bond2, atom3)

    @classmethod
    def _split_string(cls, string):
        tokens = cls._smirks_splitter(string, atoms=3)
        return cls.from_string_list(tokens)

    def __repr__(self):
        return (
            "("
            + self._atom1.__repr__()
            + ") ["
            + self._bond1.__repr__()
            + "] ("
            + self._atom2.__repr__()
            + ") ["
            + self._bond2.__repr__()
            + "] ("
            + self._atom3.__repr__()
            + ")"
        )


class DihedralGraph(ChemGraph):
    def __init__(
        self,
        atom1=None,
        bond1=None,
        atom2=None,
        bond2=None,
        atom3=None,
        bond3=None,
        atom4=None,
    ):
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
        self._fields = [
            "_atom1",
            "_bond1",
            "_atom2",
            "_bond2",
            "_atom3",
            "_bond3",
            "_atom4",
        ]

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
        return (
            "("
            + self._atom1.__repr__()
            + ") ["
            + self._bond1.__repr__()
            + "] ("
            + self._atom2.__repr__()
            + ") ["
            + self._bond2.__repr__()
            + "] ("
            + self._atom3.__repr__()
            + ") ["
            + self._bond3.__repr__()
            + "] ("
            + self._atom4.__repr__()
            + ")"
        )


class OutOfPlaneGraph(DihedralGraph):
    def __init__(self, atom1, bond1, atom2, bond2, atom3, bond3, atom4):
        super().__init__(atom1, bond1, atom2, bond2, atom3, bond3, atom4)

    def __repr__(self):
        return (
            "("
            + self._atom1.__repr__()
            + ") ["
            + self._bond1.__repr__()
            + "] ("
            + self._atom2.__repr__()
            + ") ["
            + self._bond2.__repr__()
            + "] (("
            + self._atom3.__repr__()
            + ")) ["
            + self._bond3.__repr__()
            + "] ("
            + self._atom4.__repr__()
            + ")"
        )


# import functools

# @functools.singledispatch
# def AtomType_compare_valid(a, b):
#     pass

# class AtomNumber(int):
#     pass

# class AtomBondCountHeavyRegular(int):
#     pass

# class AtomBondCountAromatic(int):
#     pass

# class AtomBondCountHydrogen(int):
#     pass

# class AtomTypeIteratorConstraint(abc.ABC):

#     def __init__(self, atom:AtomType):
#         self._atom = atom.copy()

#     @abc.abstractmethod
#     def __callable__(self) -> bool:
#         pass

# class AtomTypeConstraintBondSum(AtomTypeIteratorConstraint):
#     def __callable__(self):
#         return True

# class AtomTypeIterator():

#     def __init__(self, atom:AtomType,
#         constraints:List[AtomTypeIteratorConstraint]=[],
#         direction:bool=True):

#         self._contraints = constraints.copy()
#         self._atom = atom.copy()
#         self._direction = 1
