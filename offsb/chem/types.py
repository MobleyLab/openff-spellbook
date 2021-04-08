# /usr/bin/env python3

import abc
import itertools
import re

import numpy as np

HYDROGEN_ONE_BIT = False
HYDROGEN_ONE_BIT_SMARTS = True


class ChemTypeComparisonException(Exception):
    pass


class AtomTypeInvalidException(Exception):
    pass


class BondTypeInvalidException(Exception):
    pass


ATOM_MAXBITS = {
    "S": 118,
    "H": 5,
    "X": 8,
    "r": 10,
    "x": 8,
    "aA": 2,
    "Order": 5,
    "q": 7,  # +- 3
}


class BitVec:

    __slots__ = ["_v", "_inv", "maxbits", "fmt"]

    def __init__(self, vals=None, inv=False, maxbits=np.inf):

        self._v = np.array([False], dtype=bool)
        self._inv = inv

        if vals is None:
            pass
        elif hasattr(vals, "__iter__"):
            # assumes a list of bools
            self._v = np.array(vals, dtype=bool)
        elif np.isinf(vals):
            self._inv = np.sign(vals) ^ inv
        else:
            self[int(vals)] = True

        self.maxbits = maxbits
        self.trim()
        self.fmt = "{:>2s}"

    def bits(self, maxbits=False):
        if self._inv:
            if not maxbits:
                return np.inf
            elif self.maxbits >= len(self):
                return self.maxbits - self._v.sum()
            else:
                return len(self) - self._v.sum()
        return self._v.sum()

    def __iter__(self):
        i = 0
        for i, bit in enumerate(self._v):
            if bit ^ self._inv:
                b = BitVec()
                b[i] = True
                yield b

        i = i + 1
        while self._inv and i < self.maxbits:
            b = BitVec()
            b[i] = True
            i += 1
            yield b

    def __getitem__(self, i):

        if isinstance(i, slice):

            if i.stop is None and i.start is None and i.step is None:
                return self._v ^ self.inv

            start = 0 if i.start is None else i.start
            # end = max(self._v.shape[0] if i.stop is None else i.stop, start)
            end = i.stop

            if end is None:
                end = self.maxbits

            if np.isinf(end):
                raise IndexError(
                    "Cannot supply an infinite length array (input slice had no bounds and maxbits was inf)"
                )

            step = 1 if i.step is None else i.step

            if start >= self._v.shape[0]:
                diff = (end - start + 1) // step
                return np.full((diff,), self.inv, dtype=np.bool)

            if end > self._v.shape[0]:
                diff = (end - self._v.shape[0]) // step
                return np.concatenate(
                    (
                        self._v[start:end:step] ^ self.inv,
                        np.full((diff,), self.inv, dtype=np.bool),
                    )
                )

            return self._v[start:end:step] ^ self.inv

        elif i >= self._v.shape[0]:
            return self.inv
        else:
            return self._v[i] ^ self._inv

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

            if end > self.maxbits:
                end = self.maxbits

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
                self._v[j] = self.inv ^ v

        elif isinstance(i, int):
            if i >= self._v.shape[0]:
                if self.inv != v:
                    diff = i - self._v.shape[0] + 1
                    self._v = np.concatenate((self._v, [False] * diff))
                    self._v[i] = self.inv ^ v
            else:
                self._v[i] = self.inv ^ v
        else:
            raise Exception(
                "Using datatype {} for setitem not supported".format(type(i))
            )
        self.trim()

    def __len__(self):
        return self._v.shape[0]

    def __repr__(self):
        neg = "~" if self._inv else " "
        outstr = "".join([str(int(v)) for v in self._v[::-1]])
        return neg + self.fmt.format(outstr)

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
        if self._inv and np.isinf(self.maxbits):
            return False
        elif self._inv:
            if len(self._v) < self.maxbits:
                return False
            else:
                return np.all(self._v[: self.maxbits])
        else:
            return not np.any(self._v)

    def clear(self):
        self._v[:] = False
        self.trim()
        self.inv = False

    def copy(self):
        return BitVec(self._v, self._inv, self.maxbits)

    def all(self):
        if self.inv:
            return not self._v.any()
        else:
            return self._v.all() and self.maxbits <= self._v.shape[0]

    def any(self):
        return self.inv or self._v.any()

    def reduce(self):
        """"""
        ret = [
            2 ** b for b, i in enumerate(self._v) if i
        ]  # np.packbits(self._v), dtype=np.int32)
        # print("reduce for me is", ret, self._v)
        ret = np.sum(ret, dtype=np.int64)
        if self.inv:
            ret = -ret
        return ret

    def reduce_longest(self, o):
        """"""

        pairs = list(itertools.zip_longest(self.v, o.v, fillvalue=False))
        pairs = np.vstack(([(self.inv, o.inv)], pairs))

        suma = np.sum(
            [2 ** b for b, i in enumerate(pairs) if (i[0])], dtype=int
        )  # np.packbits(self._v), dtype=np.int32)
        sumb = np.sum(
            [2 ** b for b, i in enumerate(pairs) if (i[1])], dtype=int
        )  # np.packbits(self._v), dtype=np.int32)

        # suma, sumb = np.packbits(pairs, axis=0).sum(axis=0, dtype=np.int32)
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
        length = min(self.v.shape[0], o.v.shape[0])

        # negate bits if inverted
        a = ~self.v[:length] if self.inv else self.v[:length]
        b = ~o.v[:length] if o.inv else o.v[:length]

        # perform the logical operation
        a = fn(a, b)

        # now do the remainder, can shortcut this since the short end
        # is just comparing its inv flag

        same_length = self.v.shape[0] == o.v.shape[0]
        if not same_length:
            if length == self.v.shape[0]:
                # o is longer

                # invert if needed, then compare with others invert
                c = ~o.v[length:] if o.inv else o.v[length:]
                c = fn(c, self.inv)
            else:
                # self is longer
                c = ~self.v[length:] if self.inv else self.v[length:]
                c = fn(c, o.inv)

            a = np.concatenate((a, c))

        inv = False
        if fn(self.inv, o.inv):
            a = ~a
            inv = True

        maxbits = min(self.maxbits, o.maxbits)
        return BitVec(a, inv, maxbits=maxbits)

    def __hash__(self):
        val = (tuple(self._v), self.inv)
        # print("hash for this bitvec is", val, self)
        return hash(val)

    def __and__(self, o):
        return self._logical_op(o, np.logical_and)

    def __or__(self, o):
        return self._logical_op(o, np.logical_or)

    def __xor__(self, o):
        return self._logical_op(o, np.logical_xor)

    def __invert__(self):
        return BitVec(self._v, inv=not self._inv, maxbits=self.maxbits)

    def __add__(self, o):
        return self | o

    def __sub__(self, o):
        return self & (self ^ o)

    def __eq__(self, o):
        # a, b = self.reduce_longest(o)
        return hash(self) == hash(o)

    def __ne__(self, o):
        return not self == o


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
            # obj = getattr(self, field)
            # if type(obj) == type(ATOM_UNIVERSE):
            #     ATOM_UNIVERSE += obj
            # elif type(obj) == type(BOND_UNIVERSE):
            #     ATOM_UNIVERSE += obj
            setattr(self, field, getattr(self, field))

        self._symmetric = False

    def is_primitive(self):

        for field in self._fields:
            bv = getattr(self, field)
            if bv.bits() != 1:
                return False
        return True

    # default _is_valid
    def _is_valid(self) -> bool:
        return not self.is_null()

    def is_valid(self) -> bool:
        return self._is_valid()

    def __repr__(self):
        ret = " ".join(
            [
                "{}: {}".format(name, getattr(self, name))
                for name in self._fields
            ]
        )
        return ret

    def __iter__(self, skip_ones=False):

        """
        Absolute magic
        """

        blank = self.copy()
        blank.clear()

        for field in self._fields:
            me = getattr(self, field)
            if skip_ones and me.bits() == 1:
                continue
            bv = getattr(blank, field)
            for bit in me:
                bv += bit
                setattr(blank, field, bv)
                yield blank.copy()
                bv -= bit
                setattr(blank, field, bv)

    def _recurse_fields(self, fields: dict, pos=[]):

        if len(fields) == 0:
            return
        if len(pos) == len(fields):
            # we need to "double invert" and then save the inv flag
            # this helps with generating smarts, so we get !H0 instead of H1,H2,H3, etc
            ret = [field[i] ^ field.inv for field, i in zip(fields.values(), pos)]

            dat = {}
            # assert len(self._field_vars) == len(ret)
            for k, v in zip(fields, pos):
                dat[k] = v

            # This check is weird if there are negations since we don't limit
            # to maxbits anymore
            if all(ret):  # and self.is_valid(self.from_ints(dat)):
                yield pos.copy()
            return
            # return [field[i] for field,i in zip(fields,pos)]
        l_pos = len(pos)
        pos.append(0)
        # print("field", l+1, "/", len(fields), "has length", len(fields[l]))

        # if there is a maxbits set, do not iterate beyond it
        field_i = list(fields.values())[l_pos]
        bits = len(field_i)
        if field_i.inv:
            bits = field_i.maxbits
        for i in range(bits):
            pos[-1] = i
            # print("querying", i, "of pos", len(pos), pos)
            yield from self._recurse_fields(fields, pos=pos.copy())

    def _iter_fields(self):

        fields = {}

        inverse = {}
        for name, real_name in zip(self._fields, self._field_vars):
            field = getattr(self, name)
            # this means there is a set of bits to iterate over
            if field.any() and not field.all():
                fields[real_name] = field
                inverse[real_name] = field.inv

        vals = []

        for result in self._recurse_fields(fields, pos=[]):
            prim = {name: x for name, x in zip(fields, result)}
            prim["inv"] = inverse
            vals.append(prim)
        # vals = [{name:x for result in )}

        # hopefully this sorts! e.g. X3,X2,X1
        vals = sorted(
            vals,
            key=lambda x: tuple(int(v) for y, v in x.items() if y != "inv"),
            reverse=True,
        )
        return vals

    def clear(self):

        for field in self._fields:
            bv = getattr(self, field)
            bv.clear()

    def fill(self):

        for field in self._fields:
            bv = getattr(self, field)
            bv[:] = True

    def compact_str(self):
        ret = []
        for name in self._field_vars:
            bv = getattr(self, self._field_map[name])
            if bv.all():
                ret.append("{}*".format(name))
            elif bv.bits() > 0:
                ret.append("{}{}".format(name, bv))
        return "".join(ret)

    def all(self):

        for name in self._fields:
            vec: BitVec = getattr(self, name)
            if not vec.all():
                return False

        return True

    def any(self):

        for name in self._fields:
            vec: BitVec = getattr(self, name)
            if vec.any():
                return True

        return False

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

    def bits(self, maxbits=False, return_all=False):
        """"""

        bits = []
        for name in self._fields:
            vec: BitVec = getattr(self, name)
            bits.append(vec.bits(maxbits=maxbits))

        if return_all:
            return tuple(bits)
        else:
            return sum(bits)

    def copy(self):
        """
        Return a copy of self.
        Returns
        -------
        cls: type(cls)
           The new ChemType object
        """

        try:
            cls = self.__class__()
        except Exception as e:
            # breakpoint()
            pass

        for field in self._field_map.values():
            setattr(cls, field, getattr(self, field).copy())

        cls.inv = self.inv

        cls._field_vars = self._field_vars.copy()
        cls._fields = self._fields.copy()

        return cls

    @classmethod
    def from_dict(cls, data):
        cls = cls()
        for name in data:
            bv = data[name]
            if type(bv) is int:
                bv = BitVec(bv)
                bv.maxbits = ATOM_MAXBITS[name]
            setattr(cls, cls._field_map[name], bv)
        cls._field_vars = list(data)
        cls._fields = [cls._field_map[i] for i in data]

        return cls

    @classmethod
    def from_string(cls, string, sorted=False, unspecified_matches_all=True):
        """"""
        return cls.parse_string(string, unspecified_matches_all=True)

    @classmethod
    def from_smarts(cls, string, sorted=False, unspecified_matches_all=True):
        """"""
        return cls.parse_string(string, unspecified_matches_all=unspecified_matches_all)

    def _to_primitives(self):
        terms = [
            self.__class__.from_dict(
                {
                    k: BitVec(v, maxbits=ATOM_MAXBITS[k])
                    for k, v in zip(self._field_vars, x)
                }
            )
            for x in self._recurse_fields(
                {
                    name: getattr(self, field)
                    for name, field in zip(self._field_vars, self._fields)
                },
                pos=[],
            )
        ]
        return terms

    def _flip(self, bit: str, inv: bool) -> str:
        ret = bit
        if inv:
            ret = "1" if bit == "0" else "0"
        return ret

    def _explicit_flip(self):
        for field in self._fields:
            arr: BitVec = getattr(self, field)
            arr.explicit_flip()

    def enable(self, field):
        if field not in self._field_vars:
            status = True
        else:
            status = False
        # self._field_map[field] = self._ref_field_map[field]
        # self._field_map = {
        #     k: v for k, v in self._ref_field_map.items() if k in self._field_map
        # }

        self._fields = list(self._field_map.values())
        self._field_vars = list(self._field_map)

        return status

    def disable(self, field):
        if field in self._field_vars:
            self._fields.remove(self._field_map[field])
            self._field_vars.remove(field)
            status = True
        else:
            status = False

        return status

    def iter(self, skip_ones=False):
        yield from self.__iter__(skip_ones=skip_ones)

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

    def _permutations(self):
        """
        Generates all of the valid orderings
        of the type. Useful for comparisons,
        where we need to check if e.g A-B is
        the same as B-A or A-B.

        This default function is for all except impropers, which is either
        forward or reversed
        """
        order = tuple(range(len(self._fields)))
        return (
            order,
            order[::-1],
        )

    def align_to(self, ref):
        """
        Align the representation such that an atom by atom comparison results
        with the largest overlap.
        """

        self._check_sane_compare(ref)

        permutations = self._permutations()
        best = (None, None)
        scores = []

        for order in permutations:
            swapped = self.copy()
            swapped._fields = [swapped._fields[i] for i in order]
            ans = swapped - ref
            bits = ans.bits(maxbits=True, return_all=True)
            if best[1] is None or bits < best[1]:
                best = (order, bits)
            scores.append((order, bits, ans))

        if best[1] is not None:
            reordered = [getattr(self, self._fields[i]) for i in best[0]]
            for i, val in enumerate(reordered):
                setattr(self, self._fields[i], val)

    def drop(self, other):
        self._check_sane_compare(other)
        diff: ChemType = other & self
        if diff.reduce() == 0:
            return self
        ans = self.copy()
        for field_name in self._fields:
            field: ChemType = getattr(diff, field_name)
            if field.is_null():
                continue
            setattr(ans, field_name, field)
        return ans

    def _check_sane_compare(self, o):
        if type(self) != type(o):
            raise ChemTypeComparisonException("ChemType operations must use same type")

    def _trim(self, dat):
        "Remove leading zeros"

        for field in self._fields:
            vec: ChemType = getattr(self, field)
            vec.trim()

    def _dispatch_op(self, fn):

        args = [getattr(self, field) for field in self._fields]
        ret = fn(*args, inv=self.inv)
        ret = [self._trim(r) for r in ret]
        return self.from_data_string(*ret)

    def __and__(self, o):
        # bitwise and (intersection)

        self._check_sane_compare(o)

        ret = {}
        all_names = set(self._field_vars + o._field_vars)
        for name_a, field_a in zip(self._field_vars, self._fields):
            for name_b, field_b in zip(o._field_vars, o._fields):
                if name_a == name_b:
                    a_vec = getattr(self, field_a)
                    b_vec = getattr(o, field_b)
                    ret[name_a] = a_vec & b_vec
                    all_names.remove(name_a)
                    break
        for name in all_names:
            ret[name] = BitVec()

        return self.from_dict(ret)

    def __or__(self, o):
        # bitwise or (union)

        self._check_sane_compare(o)

        ret = {}
        for name_a, field_a in zip(self._field_vars, self._fields):
            for name_b, field_b in zip(o._field_vars, o._fields):
                if name_a == name_b:
                    a_vec = getattr(self, field_a)
                    b_vec = getattr(o, field_b)
                    ret[name_a] = a_vec | b_vec
                    break
            if name_a not in ret:
                a_vec = getattr(self, field_a)
                ret[name_a] = a_vec

        for name_b, field_b in zip(o._field_vars, o._fields):
            if name_b not in ret:
                b_vec = getattr(o, field_b)
                ret[name_b] = b_vec

        return self.from_dict(ret)

    def __xor__(self, o):
        # bitwise xor

        self._check_sane_compare(o)

        ret = {}
        for name_a, field_a in zip(self._field_vars, self._fields):
            for name_b, field_b in zip(o._field_vars, o._fields):
                if name_a == name_b:
                    a_vec = getattr(self, field_a)
                    b_vec = getattr(o, field_b)
                    ret[name_a] = a_vec ^ b_vec
                    break
            if name_a not in ret:
                a_vec = getattr(self, field_a)
                ret[name_a] = a_vec

        for name_b, field_b in zip(o._field_vars, o._fields):
            if name_b not in ret:
                b_vec = getattr(o, field_b)
                ret[name_b] = b_vec

        return self.from_dict(ret)

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

        # note that we cannot compose this with XOR, since the
        # __init__ might reorder the intermediate bit vector
        # before it is ANDed, causing the bit to possibly apply
        # to the wrong atom
        self._check_sane_compare(o)

        # do some potentially wicky wacky magic to make downstream easier
        # o is in self but they are misordered, we would potentially miss
        # the more logical calculation

        # for example, if we have a 01 10 and a 01 01 and we substract 01 00,
        # then we mean to take select the second, but would be missed without
        # explicitly checking. Thus, as a compromised, try to flip things
        # if its not found the first time. Note that this will probably lead
        # to some complicated situations when manipulations become more complex

        # now that we have an align function, try to do the normal thing

        # pairs = zip(self._fields, o._fields)

        # this means o is not aligned with self, so we try to get a better
        # alignment by flipping it
        # if self & o != o and self._symmetric and o._symmetric:
        #     # TODO get this working for impropers
        #     pairs = zip(self._fields, o._fields[::-1])

        ans = self ^ o
        return self & ans

    def __contains__(self, o):
        self._check_sane_compare(o)

        ret = True
        for a_field, o_field in zip(self._fields, o._fields):
            a_vec: ChemType = getattr(self, a_field)
            o_vec: ChemType = getattr(o, o_field)
            if (o_vec - a_vec).any():
                ret = False
                break
        if not ret and self._symmetric:
            ret = True
            for a_field, o_field in zip(self._fields, o._fields[::-1]):
                a_vec: ChemType = getattr(self, a_field)
                o_vec: ChemType = getattr(o, o_field)
                if (o_vec - a_vec).any():
                    return False
        return ret

        # return (self - o).reduce() > 0

    def __hash__(self):
        fields = tuple(
            [
                hash(getattr(self, field))
                for field in self._fields
            ]
        )
        if self._symmetric:
            swapped = tuple(
                [
                    hash(getattr(self, field))
                    for field in self._fields[::-1]
                ]
            )
            return max(hash(fields), hash(swapped))

        # print("For type", type(self), "fields is", fields, self._fields)
        return hash(fields)

    def __eq__(self, o):
        return hash(self) == hash(o)

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
        return not self == o


class AtomType(ChemType):
    """"""

    _field_map = {
        "S": "_symbol",
        "H": "_H",
        "X": "_X",
        "x": "_x",
        "r": "_r",
        "aA": "_aA",
        "q": "_q",
    }

    _ref_field_map = {
        "S": "_symbol",
        "H": "_H",
        "X": "_X",
        "x": "_x",
        "r": "_r",
        "aA": "_aA",
        "q": "_q",
    }

    def __init__(
        self,
        symbol: BitVec = None,
        X: BitVec = None,
        x: BitVec = None,
        H: BitVec = None,
        r: BitVec = None,
        aA: BitVec = None,
        q: BitVec = None,
        inv: bool = False,
    ):
        """"""

        self._field_vars = ["S", "H", "X", "x", "r", "aA", "q"]
        self._fields = ["_symbol", "_H", "_X", "_x", "_r", "_aA", "_q"]

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

        if q is None:
            q = BitVec()
        self._q = q

        if HYDROGEN_ONE_BIT and symbol is not None and symbol[1] and symbol.bits() == 1:
            self._field_vars = ["S"]
            self._fields = ["_symbol"]

        super().__init__(inv=inv)

        self._symmetric = False

    @classmethod
    def from_data(
        cls,
        symbol: BitVec,
        H: BitVec,
        X: BitVec,
        x: BitVec,
        r: BitVec,
        aA: BitVec,
        q: BitVec,
        inv: bool = False,
    ):
        """"""

        return cls(symbol=symbol, X=X, H=H, x=x, r=r, aA=aA, q=q, inv=inv)

    @classmethod
    def from_dict(cls, data):
        cls = cls()
        for name in data:
            bv = data[name]
            if type(bv) is int:
                bv = BitVec(bv)

            bv.maxbits = ATOM_MAXBITS[name]
            setattr(cls, cls._field_map[name], bv)
        cls._field_vars = list(data)
        cls._fields = [cls._field_map[i] for i in data]

        symbol = cls._symbol
        if HYDROGEN_ONE_BIT and symbol is not None and symbol[1] and symbol.bits() == 1:
            cls._field_vars = ["S"]
            cls._fields = ["_symbol"]

        return cls

    @classmethod
    def from_ints(cls, data, inv: bool = False):
        """"""

        """
        None means null (0)
        inf means all (~0)
        -1 means not first bit on (~1)
        """

        cls = cls()

        for name, field in zip(cls._field_vars, cls._fields):
            v = data.get(name, None)
            setattr(cls, field, BitVec(v, inv=v < 0))

        # vec = BitVec()
        # i = data.get("X", False)
        # if i == False:
        #     vec.inv = True
        # else:
        #     vec[i] = True
        # cls._X = vec.copy()

        # vec = BitVec()
        # i = data.get("x", False)
        # if i == False:
        #     vec.inv = True
        # else:
        #     vec[i] = True
        # cls._x = vec.copy()

        # vec = BitVec()
        # i = data.get("H", False)
        # if i == False:
        #     vec.inv = True
        # else:
        #     vec[i] = True
        # cls._H = vec.copy()

        # vec = BitVec()
        # i = data.get("r", False)
        # if i == False:
        #     vec.inv = True
        # else:
        #     vec[i] = True
        # cls._r = vec.copy()

        # vec = BitVec()
        # i = data.get("aA", False)
        # if i == False:
        #     vec.inv = True
        # else:
        #     vec[i] = True
        # cls._aA = vec.copy()

        return cls

    @classmethod
    def from_string_list(cls, string_list, sorted=False):
        atom1 = cls.from_string(string_list[0])
        return atom1

    @classmethod
    def parse_string(self, string, unspecified_matches_all=True):
        """"""

        # splits a single atom record, assumes a single primitive for now
        # therefore things like wildcards or ORs not supported

        global ATOM_UNIVERSE

        # strip the [] brackets
        # atoms = string[1:-1]

        atoms = string.split(",")

        # hack? I really don't want to write a general parser
        # another hack: turn on anything not null at the end
        turn_on_null = False
        if len(atoms) > 1:
            unspecified_matches_all = False
            turn_on_null = True

        self = AtomType()
        group = None
        for atom in atoms:

            # these can be safely removed, since they are always impied??
            atom = atom.replace(";", "")
            # The symbol number
            pat = re.compile(r"(!?)#([1-9][0-9]*)")
            ret = pat.search(atom)
            symbol = BitVec()
            symbol.maxbits = ATOM_MAXBITS["S"]
            if ret:
                symbol[int(ret.group(2))] = True
                if ret.group(1) == "!":
                    symbol = ~symbol
            elif atom.startswith("*"):
                symbol[:] = True
            elif unspecified_matches_all:
                symbol[:] = True

            # The number of bonds
            pat = re.compile(r"(!?)X([0-9][1-9]*)")
            ret = pat.search(atom)
            X = BitVec()
            X.maxbits = ATOM_MAXBITS["X"]  # consider total bonds of 0-6
            if ret:
                X[int(ret.group(2))] = True
                if ret.group(1) == "!":
                    X = ~X
            elif unspecified_matches_all:
                X[:] = True

            # The number of bonds which are aromatic
            pat = re.compile(r"(!?)x([0-9][1-9]*)")
            ret = pat.search(atom)
            x = BitVec()
            x.maxbits = ATOM_MAXBITS["x"]  # consider aromatic bonds of 0-4
            if ret:
                x[int(ret.group(2))] = True
                if ret.group(1) == "!":
                    x = ~x
            elif unspecified_matches_all:
                x[:] = True

            # The number of bonds which are to hydrogen
            pat = re.compile(r"(!?)H([0-9][1-9]*)")
            ret = pat.search(atom)
            H = BitVec()
            H.maxbits = ATOM_MAXBITS["H"]  # consider H of 0-4
            if ret:
                H[int(ret.group(2))] = True
                if ret.group(1) == "!":
                    H = ~H
            elif unspecified_matches_all:
                H[:] = True

            # The size of the ring membership
            pat = re.compile(r"(!?)r([0-9]?)")
            ret = pat.search(atom)
            r = BitVec()

            r.maxbits = ATOM_MAXBITS["r"]  # consider rings of 0-8 (skip 1 and 2)
            if ret:
                if ret.group(2) != "":
                    # order is 0:0 1:None 2:None 3:1 4:2 2:r3 r4
                    # since r1 and r2 are useless, r3 maps to second bit
                    r[int(ret.group(2)) - 2] = True
                else:
                    # this means just r was given, so ~1
                    r[1:] = True
                if ret.group(1) == "!":
                    r = ~r
            elif unspecified_matches_all:
                r[:] = True

            # Whether the atom is considered aromatic
            pat = re.compile(r"(!?)([aA])")
            ret = pat.search(atom)
            aA = BitVec()
            aA.maxbits = ATOM_MAXBITS["aA"]
            if ret:
                if ret.group(2) == "a":
                    aA[1] = True
                else:
                    aA[0] = True
                if ret.group(1) == "!":
                    aA = ~aA
            elif unspecified_matches_all:
                aA[:] = True

            # charge
            q = BitVec()
            q.maxbits = ATOM_MAXBITS["q"]
            empty = True

            # pat = re.compile(r"(+[0-9]?|++*|-[0-9]?|--?)")
            number = re.compile(r"(!?)(\+|-)([0-9]+)")
            ret = number.search(atom)
            val = None
            if ret:

                sign = 0 if ret.group(2)[0] == "+" else 1
                val = int(ret.group(3))
                empty = False
            else:
                repeat = re.compile(r"(!?)(\+\+*|--*)")
                ret = repeat.search(atom)
                if ret:
                    sign = 0 if ret.group(2)[0] == "+" else 1
                    val = len(ret.group(2))
                    empty = False

            if not empty:
                if val == 0:
                    q[0] = True
                else:
                    # positive is odd, negative is even
                    q[(val - 1) * 2 + sign + 1] = True
                if ret.group(1) == "!":
                    q = ~q
            elif unspecified_matches_all:
                q[:] = True

            # # sanitize: assume hydrogen is a single bonded atom
            if HYDROGEN_ONE_BIT and symbol[1] and symbol.bits() == 1:
                H.clear()
                # H[0] = True
                # H[1:] = False

                H.clear()
                # X[0] = False
                # X[2:] = False

                x.clear()
                # x[2:] = False

                r.clear()
                # r[1:] = False

                aA.clear()
                # aA[:] = False
                # aA[0] = True

                q.clear()

                obj = self.from_dict({"S": symbol})

            else:
                obj = self.from_data(symbol, H, X, x, r, aA, q)
            if group is None:
                group = obj
            else:
                group += obj

        ATOM_UNIVERSE += self

        if turn_on_null:
            for field in group._fields:
                field = getattr(group, field)
                if field.is_null():
                    field[:] = True

        return group

    def bit_str(self):
        return "{} {} {} {} {} {}".format(
            self._symbol, self._H, self._X, self._x, self._r, self._aA
        )

    def to_smarts(self, tag=None, separate=False, universe=None):
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

        # the "regular" terms that can be done easily
        mapping = {"S": "#", "H": "H", "X": "X", "x": "x"}

        if universe is not None:
            self = self.copy()
            o = self ^ universe
            for field in o._fields:
                bv = getattr(self, field)
                obv = getattr(o, field)
                if obv.is_null():
                    bv[:] = True

        vals = self._iter_fields()

        for field in mapping:
            term_list = set()
            inv = False
            for result in sorted(vals, key=lambda x: x.get(field, 1), reverse=True):

                val_str = ""
                val = result.get(field, None)
                joiner = ","
                if val is not None:
                    val_str = "{}{}".format(mapping[field], val)
                    inv = result["inv"][field]
                    if inv:
                        val_str = "!" + val_str
                        joiner = ""

                    # hack: don't print !X0 as it is not useful right now
                    if val_str == "!X0":
                        continue
                    if "#0" in val_str:
                        continue

                    term_list.add(val_str)
            if len(term_list):
                term = joiner.join(term_list)
                # if inv:
                #     term = "!("+term+")"
                terms.append(term)

        inv = False
        term_list = set()
        for result in sorted(vals, key=lambda x: x.get("r", 1), reverse=True):
            field = "r"
            ring_str = ""
            ring_val = result.get(field, None)
            joiner = ","
            if ring_val is not None:
                inv = result["inv"][field]
                if inv:
                    joiner = ""
                if not inv:
                    ring_str = "!r" if ring_val == 0 else "r{}".format(ring_val + 2)
                elif ring_val > 0:
                    ring_str = "!r{}".format(ring_val + 2)
                else:
                    ring_str = "r"
                # inv = result['inv'][field]
                # ring_str = "!r" if ring_val == 0 else "r{}".format(ring_val + 2)
                # elif ring_val > 0:
                #     ring_str = "!r{}".format(ring_val + 2)
                # else:
                #     ring_str = "r"
                term_list.add(ring_str)

        if len(term_list):
            term = joiner.join(term_list)
            terms.append(term)

        inv = False
        term_list = set()
        for result in vals:
            joiner = ","
            field = "aA"
            aA = ""
            aA_val = result.get(field, None)
            if aA_val != None:
                inv = result["inv"][field]
                # if inv:
                #     joiner = ""
                aA = "A" if (aA_val ^ inv) == 0 else "a"
                # aA = "a" if aA_val else "A"
                term_list.add(aA)

        # only worth printing if one is on
        if len(term_list) == 1:
            term = joiner.join(term_list)
            terms.append(term)

        inv = False
        term_list = set()
        for result in sorted(vals, key=lambda x: x.get("q", 1), reverse=True):
            field = "q"
            q_str = ""
            q_val = result.get(field, None)
            joiner = ","
            not_str = ""
            if q_val is not None:
                inv = result["inv"][field]
                if inv:
                    joiner = ""
                    not_str = "!"
                if q_val == 0:
                    q_str = "+0"
                elif q_val % 2:
                    q_str = "+{}".format(q_val // 2 + 1)
                else:
                    q_str = "-{}".format(q_val // 2)

                term_list.add(not_str + q_str)

        if len(term_list):
            term = joiner.join(term_list)
            terms.append(term)

        if tag is True:
            tag = 1
        tag = "" if tag is None else ":" + str(int(tag))
        if len(terms) > 0:
            sep = ";" if separate else ""
            smarts = sep.join(terms)
        else:
            smarts = "*"
        # if len(smarts) == 0:
        #     breakpoint()
        if self._symbol[1] and self._symbol.bits() == 1 and HYDROGEN_ONE_BIT_SMARTS:
            smarts = "#1"
        return "[" + smarts + tag + "]"

    def to_primitives(self):
        """
        Converts the representation to an enumeration of all possible
        primitive types that are valid.
        """
        if self.is_null():
            raise AtomTypeInvalidException("Does not define any primitive")

        return self._to_primitives()

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

        if HYDROGEN_ONE_BIT and self._symbol[1] and self._symbol.bits() == 1:
            return False
        else:
            return super().is_null()

    def _is_valid(self) -> bool:

        sym = np.inf if self._symbol.inv else len(self._symbol) - 1
        H = np.inf if self._H.inv else len(self._H) - 1
        x = np.inf if self._x.inv else len(self._x) - 1
        X = np.inf if self._X.inv else len(self._X) - 1
        r = np.inf if self._r.inv else len(self._r) - 1

        # If something is inf, then just
        # let it go... it means we have all bits
        # so we can satisfy it somehow down the road
        prim = self.is_primitive()

        # if X0 or X1, then can't be in a ring
        if prim and (not np.isinf(X + r) and X < 2 and r > 1):
            return False

        # The sum of H and x must be <= X
        if prim and not np.isinf(X + x + H) and X < x + H:
            return False

        # if self._aA[0] and self._aA[1]:
        #     return False

        # Everything else is acceptable?
        return not self.is_null()


class BondType(ChemType):
    """"""

    _field_map = {"Order": "_order", "aA": "_aA"}

    def __init__(self, order: BitVec = None, aA: BitVec = None, inv: bool = False):
        """"""

        self._field_vars = ["Order", "aA"]
        self._fields = ["_order", "_aA"]
        if order is None:
            order = BitVec()
        if aA is None:
            aA = BitVec(maxbits=2)
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

        for name, field in zip(cls._field_vars, cls._fields):
            v = data.get(name, None)
            setattr(cls, field, BitVec(v, inv=v < 0))

        return cls

    @classmethod
    def parse_string(self, string, unspecified_matches_all=True):
        """
        splits a single atom record, assumes a single primitive for now
        therefore things like wildcards or ORs not supported
        """

        global BOND_UNIVERSE

        self = BondType()
        for string in string.split(","):
            order = BitVec()
            order.maxbits = 6
            aA = BitVec()
            aA.maxbits = 2

            aro = False
            aro_idx = 3
            if "@" in string:
                aro = True
                pat = re.compile(r"(!?)(.)[;&]*(!?@)")

            else:
                pat = re.compile(r"(!?)(.)")

            ret = pat.search(string)

            if ret:
                sym = ret.group(2)
                if sym == "~":
                    order[:] = True
                elif sym == "-":
                    order[1] = True
                elif sym == "=":
                    order[2] = True
                elif sym == "#":
                    order[3] = True
                elif sym == ":":
                    order[4] = True
                else:
                    # This should be for []@[] style bonds
                    order[:] = True
                    aro = True
                    aro_idx = 2
                if ret.group(1) == "!" and not order.all():
                    order = ~order

                if aro:
                    n_g = len(ret.groups())
                    aA[1] = True if n_g == 3 and ret.group(aro_idx) == "@" else False
                    aA[0] = True if n_g == 3 and ret.group(aro_idx) == "!@" else False
                if aA.reduce() == 0 and unspecified_matches_all:
                    aA[:] = True

            else:  # nothing always means single bond, any aromaticity
                order[1] = True
                aA[:] = True

            self += self.from_data(order, aA)

        BOND_UNIVERSE += self
        return self

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

    def _is_valid(self) -> bool:

        if not super()._is_valid():
            return False

        # if self._aA[0] and self._aA[1]:
        #     return False

        # Everything else is acceptable?
        return not self.is_null()

    def to_smarts(self, separate=False, universe=None):
        """
        Converts the representation to an enumeration of all possible
        primitive types that are valid.
        """

        if not self.is_valid():
            raise BondTypeInvalidException()

        if universe is not None:
            self = self.copy()
            o = self ^ universe
            for field in o._fields:
                bv = getattr(self, field)
                obv = getattr(o, field)
                if obv.is_null():
                    bv[:] = True

        bond_lookup = {1: "-", 2: "=", 3: "#", 4: ":"}

        terms = list()

        vals = self._iter_fields()

        # bond order
        bond = ""
        joiner = ","
        term_list = set()
        for result in sorted(vals, key=lambda x: x.get("Order", 1), reverse=True):
            bond_val = result.get("Order", None)
            if bond_val is not None:

                # Just in case, if the 0 bit is on (no bond), skip it
                # If we actually want to have no bond, we have failed upstream
                # since we shouldn't have tried to call this on a bond
                inv = result["inv"]["Order"]
                if bond_val == 0:
                    continue

                if bond_val > 4:
                    continue
                bond = bond_lookup[bond_val]
                if inv:
                    bond = "!" + bond
                    joiner = ""
                term_list.add(bond)

        if len(term_list):
            term = joiner.join(term_list)
            terms.append(term)

        # in a ring?
        term_list = set()
        joiner = ","
        for result in vals:

            aA = ""
            aA_val = result.get("aA", None)
            if aA_val is not None:
                inv = result["inv"]["aA"]
                aA = "!@" if (aA_val ^ inv) == 0 else "@"
                term_list.add(aA)

        # only worth printing if one is on
        if len(term_list) == 1:
            term = joiner.join(term_list)
            terms.append(term)

        smarts = "~"
        sep = ";" if separate else ""
        if len(terms) > 0:
            smarts = sep.join(terms)

        return smarts

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

        return self._to_primitives()


class ChemGraph(ChemType, abc.ABC):
    def __init__(self):
        super().__init__()
        self._symmetric = False

    def is_null(self):

        for field in self._fields:
            a = getattr(self, field)
            if a.is_null():
                return True

        return False

    def _is_valid(self):

        for field in self._fields:
            a = getattr(self, field)
            if not a.is_valid():
                return False

        return True

    def is_primitive(self):

        for field in self._fields:
            a = getattr(self, field)
            if not a.is_primitive():
                return False

        return True

    def reduce(self):

        reduce = 0
        for field in self._fields:
            a = getattr(self, field)
            reduce += a.reduce()
        return reduce

    def to_smarts(self, tag=True, atom_universe=None, bond_universe=None):
        smarts = ""
        for field in self._fields:
            chemtype = getattr(self, field)
            u = atom_universe if type(chemtype) is AtomType else bond_universe
            smarts += chemtype.to_smarts(tag=tag, universe=u)

        return smarts

    @classmethod
    def from_string(cls, string, sorted=False):
        return cls._split_string(string, sorted=sorted)

    @classmethod
    def from_smarts(cls, string, sorted=False):
        return cls.from_string(string, sorted=sorted)

    @classmethod
    def _smirks_splitter(cls, smirks, atoms):
        import re

        atom = r"(\[[^[.]*:?[0-9]*\])"
        bond = r"([^[.]*)"
        extra = r"(.*$)"
        smirks = smirks.strip("()")
        if atoms <= 0:
            return tuple()
        pat_str = atom
        for i in range(1, atoms):
            pat_str += bond + atom
        pat_str += extra

        pat = re.compile(pat_str)
        ret = pat.match(smirks)
        ret = ret.groups()
        if ret[-1] == "":
            ret = ret[:-1]
        return ret

    def _check_sane_compare(self, o):
        if len(self._fields) != len(o._fields):
            raise ChemTypeComparisonException
        for field_a, field_b in zip(self._fields, o._fields):
            try:
                a = getattr(self, field_a)
                b = getattr(o, field_b)
            except AttributeError:
                # breakpoint()
                raise ChemTypeComparisonException
            a._check_sane_compare(b)

    def compact_str(self):
        ret = []
        for field in self._fields:
            obj = getattr(self, field)
            ret.append("[" + obj.compact_str() + "]")
        return "".join(ret)

    def __and__(self, o):
        # bitwise and (intersection)

        self._check_sane_compare(o)
        ret_fields = []
        for field_a, field_b in zip(self._fields, o._fields):
            a = getattr(self, field_a)
            b = getattr(o, field_b)
            ret_fields.append(a & b)
            # if self.inv or o.inv:
            #     self.inv = True

        return self.from_data(*ret_fields)

    def __or__(self, o):
        # bitwise or (intersection)
        self._check_sane_compare(o)
        ret_fields = []
        for field_a, field_b in zip(self._fields, o._fields):
            a = getattr(self, field_a)
            b = getattr(o, field_b)
            ret_fields.append(a | b)
            # if self.inv or o.inv:
            #     self.inv = True

        return self.from_data(*ret_fields)

    def __xor__(self, o):
        # bitwise xor
        self._check_sane_compare(o)
        ret_fields = []
        for field_a, field_b in zip(self._fields, o._fields):
            a = getattr(self, field_a)
            b = getattr(o, field_b)
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

    # def __add__(self, o):
    #     # a + b is union
    #     return self.__or__(o)

    # def __sub__(self, o):
    #     # a - b is a marginal, note that a - b != b - a
    #     a = self.__xor__(o)
    #     return self & a

    # def __eq__(self, o):
    #     # a, b = self.reduce_longest(o)
    #     return hash(self) == hash(o)

    # def __lt__(self, o):
    #     a, b = self.reduce_longest(o)
    #     return a < b

    # def __gt__(self, o):
    #     a, b = self.reduce_longest(o)
    #     return a > b

    # def __le__(self, o):
    #     a, b = self.reduce_longest(o)
    #     return a <= b

    # def __ge__(self, o):
    #     a, b = self.reduce_longest(o)
    #     return a >= b

    # def __ne__(self, o):
    #     a, b = self.reduce_longest(o)
    #     return a != b


class BondGroup(ChemGraph):
    def __init__(self, atom1=None, bond=None, atom2=None, sorted=False):
        if atom1 is None:
            atom1 = AtomType()
        if bond is None:
            bond = BondType()
        if atom2 is None:
            atom2 = AtomType()
        self._atom1 = atom1 + atom2
        self._atom1._X[0] = 0
        self._atom1._symbol[0] = 0
        bond._order[0] = False
        self._bond = bond
        self._fields = ["_atom1", "_bond"]
        super().__init__()

    @classmethod
    def from_data(cls, atom1, bond, atom2=None):
        return cls(atom1, bond, atom2)

    @classmethod
    def from_string_list(cls, string_list, sorted=False):

        atom1 = AtomType.from_string(string_list[0])
        bond1 = BondType.from_string(string_list[1])

        atom2 = None
        if len(string_list) == 3:
            atom2 = AtomType.from_string(string_list[2])

        if len(string_list) == 4:
            raise ValueError("Too many values to unpack, expected 2 or 3")

        return cls(atom1, bond1, atom2, sorted=False)

    @classmethod
    def _split_string(cls, string, sorted=False):
        tokens = cls._smirks_splitter(string, atoms=2)
        return cls.from_string_list(tokens, sorted=sorted)

    def to_smarts(self, tag=True):

        if tag:
            return (
                self._atom1.to_smarts(1)
                + self._bond.to_smarts()
                + self._atom1.to_smarts(2)
            )
        else:
            return (
                self._atom1.to_smarts()
                + self._bond.to_smarts()
                + self._atom1.to_smarts()
            )

    def to_graph(self):
        prims = []
        for field in self._fields:
            obj = getattr(self, field)
            # prims.append(set([x.to_smarts() for x in obj.to_primitives()]))
            prims.append(set(obj.to_primitives()))

        graph = BondGraph()

        for a1 in prims[0]:
            for bnd in prims[1]:
                for a2 in prims[0]:
                    if a2 < a1:
                        bg = BondGraph(a2, bnd, a1)
                    else:
                        bg = BondGraph(a1, bnd, a2)

                    graph += bg

        return graph

    def drop(self, other):

        graph = self.to_graph()
        graph._atom1 = graph._atom1.drop(other._atom1)
        graph._bond = graph._bond.drop(other._bond)
        return graph

        # if hasattr(other, "to_primitives"):
        #     prims = other.to_primitives()
        # else:
        #     prims = list(other)
        # return [x for x in self.to_primitives() if x not in prims]

        # # summing will never work
        # return functools.reduce(
        #     lambda x, y: x + y, [x for x in self.to_primitives() if x not in prims]
        # )

    @classmethod
    def primitive_sort(self, prims):
        import tqdm

        ret = set()
        for a1 in tqdm.tqdm(
            prims[0],
            total=len(prims[0]),
            desc="prims1",
            ncols=80,
            disable=np.prod(list(map(len, prims))) < 10000,
        ):
            for bnd in prims[1]:
                for a2 in prims[0]:

                    found = False
                    for existing in ret:
                        if existing._atom1 == a1:
                            graph = BondGraph(a1, bnd, a2)
                            found = True
                            break
                        elif existing._atom1 == a2:
                            graph = BondGraph(a1, bnd, a2)
                            found = True
                            break
                    if a2 < a1:
                        graph = BondGraph(a2, bnd, a1)
                    else:
                        graph = BondGraph(a1, bnd, a2)

                    ret.add(graph)

    def to_primitives(self):

        import tqdm

        prims = []

        for field in tqdm.tqdm(
            self._fields, total=len(self._fields), desc="types", disable=True
        ):
            obj = getattr(self, field)
            prims.append(set(obj.to_primitives()))

        ret = set()
        for a1 in tqdm.tqdm(
            prims[0],
            total=len(prims[0]),
            desc="prims1",
            ncols=80,
            disable=np.prod(list(map(len, prims))) < 10000,
        ):
            for bnd in prims[1]:
                for a2 in prims[0]:

                    found = False
                    for existing in ret:
                        if existing._atom1 == a1:
                            graph = BondGraph(a1, bnd, a2)
                            found = True
                            break
                        elif existing._atom1 == a2:
                            graph = BondGraph(a2, bnd, a1)
                            found = True
                            break
                    if not found:
                        if a2 > a1:
                            graph = BondGraph(a2, bnd, a1)
                        else:
                            graph = BondGraph(a1, bnd, a2)

                    if graph.is_primitive():
                        ret.add(graph)

        return list(ret)

    def __repr__(self):
        return "(" + self._atom1.__repr__() + ") [" + self._bond.__repr__() + "]"


class BondGraph(ChemGraph):
    def __init__(self, atom1=None, bond=None, atom2=None, sorted=False):
        if atom1 is None:
            atom1 = AtomType()
        if bond is None:
            bond = BondType()
        if atom2 is None:
            atom2 = AtomType()
        atom1._symbol[0] = 0
        atom1._X[0] = 0
        bond._order[0] = 0
        self._atom1 = atom1
        self._bond = bond
        self._atom2 = atom2
        self._fields = ["_atom1", "_bond", "_atom2"]

        if sorted and atom2 < atom1:
            atom2, atom1 = atom1, atom2

        super().__init__()

        self._symmetric = True

    @classmethod
    def from_data(cls, atom1, bond, atom2):
        return cls(atom1, bond, atom2)

    @classmethod
    def from_string_list(cls, string_list, sorted=False):
        atom1 = AtomType.from_string(string_list[0])
        bond1 = BondType.from_string(string_list[1])
        atom2 = AtomType.from_string(string_list[2])
        if len(string_list) > 3:
            raise NotImplementedError(
                "The SMARTS pattern supplied has not been implemented"
            )
        return cls(atom1, bond1, atom2, sorted=sorted)

    @classmethod
    def _split_string(cls, string, sorted=False):
        tokens = cls._smirks_splitter(string, atoms=2)
        return cls.from_string_list(tokens, sorted=sorted)

    def compiled_smarts(self):
        """
        take a list of bonds and try to produce the smallest smarts string possible
        """
        pass

    def to_smarts(self, tag=True, atom_universe=None, bond_universe=None):

        if tag:
            return (
                self._atom1.to_smarts(1, universe=atom_universe)
                + self._bond.to_smarts(universe=bond_universe)
                + self._atom2.to_smarts(2, universe=atom_universe)
            )
        else:
            return (
                self._atom1.to_smarts(universe=atom_universe)
                + self._bond.to_smarts(universe=bond_universe)
                + self._atom2.to_smarts(universe=atom_universe)
            )

    def drop(self, other):

        graph = self.copy()
        graph._atom1 = graph._atom1.drop(other._atom1)
        graph._bond = graph._bond.drop(other._bond)

        if type(other) == type(self):
            graph._atom2 = graph._atom2.drop(other._atom2)

        return graph

    def _is_valid(self) -> bool:
        if self.is_null():
            return False

        return True

    def is_primitive(self):

        if super().is_null():
            return False

        if not super().is_primitive():
            return False

        return True

        # Skipping these for now; do we want unbonded associations?
        if self._atom1._X._v[0]:
            return False

        if self._atom2._X._v[0]:
            return False

        if self._atom1._H._v[0] and self._atom2._symbol._v[1]:
            return False

        if self._atom2._H._v[0] and self._atom1._symbol._v[1]:
            return False

        return True

    def cluster(self, primitives=None):
        if primitives is None:
            primitives = self.to_primitives()

        groups = {}
        for prim in primitives:
            a1, b, a2 = prim._atom1, prim._bond, prim._atom2
            if a1 < a2:
                a2, a1 = a1, a2
            if a1 not in groups:
                groups[a1] = []
            groups[a1].extend([b, a2])

        return groups

    def to_primitives(self):
        import tqdm

        prims = []
        for field in tqdm.tqdm(self._fields, total=len(self._fields), desc="types"):
            obj = getattr(self, field)
            prims.append(set(obj.to_primitives()))

        ret = set()
        for a1 in tqdm.tqdm(
            prims[0],
            total=len(prims[0]),
            desc="prims",
            disable=np.prod(list(map(len, prims))) < 10000,
        ):
            for bnd in prims[1]:
                for a2 in prims[2]:

                    found = False
                    for existing in ret:
                        if existing._atom1 == a1:
                            graph = BondGraph(a1, bnd, a2)
                            found = True
                            break
                        elif existing._atom1 == a2:
                            graph = BondGraph(a2, bnd, a1)
                            found = True
                            break
                    if not found:
                        if a2 > a1:
                            graph = BondGraph(a2, bnd, a1)
                        else:
                            graph = BondGraph(a1, bnd, a2)

                    if graph.is_primitive():
                        ret.add(graph)

        return list(ret)

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


class AngleGroup(ChemGraph):
    def __init__(
        self, atom1=None, bond1=None, atom2=None, bond2=None, atom3=None, sorted=False
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
        self._atom1 = atom1 + atom3
        self._bond1 = bond1 + bond2
        self._atom2 = atom2

        self._fields = ["_atom1", "_bond1", "_atom2"]
        super().__init__()

    @classmethod
    def from_data(cls, atom1, bond1, atom2, bond2=None, atom3=None):
        return cls(atom1, bond1, atom2, bond2, atom3)

    @classmethod
    def from_string_list(cls, string_list, sorted=False):
        atom1 = AtomType.from_string(string_list[0])
        bond1 = BondType.from_string(string_list[1])
        atom2 = AtomType.from_string(string_list[2])
        bond2 = None
        atom3 = None
        if len(string_list) > 3:
            bond2 = BondType.from_string(string_list[3])
            atom3 = AtomType.from_string(string_list[4])
        if len(string_list) > 5:
            raise NotImplementedError(
                "The SMARTS pattern supplied has not been implemented"
            )
        return cls(atom1, bond1, atom2, bond2, atom3, sorted=False)

    @classmethod
    def _split_string(cls, string, sorted=False):
        tokens = cls._smirks_splitter(string, atoms=3)
        return cls.from_string_list(tokens, sorted=sorted)

    def __repr__(self):
        return (
            "("
            + self._atom1.__repr__()
            + ") ["
            + self._bond1.__repr__()
            + "] ("
            + self._atom2.__repr__()
            + ")"
        )

    def to_primitives(self):
        prims = []
        for field in self._fields:
            obj = getattr(self, field)
            prims.append(set(obj.to_primitives()))

        ret = set()

        for a1 in prims[0]:
            for bnd1 in prims[1]:
                for a2 in prims[2]:
                    for bnd2 in prims[1]:
                        if bnd2 < bnd1:
                            continue
                        for a3 in prims[0]:
                            if a3 < a1:
                                continue
                            prim = a1 + bnd1 + a2 + bnd2 + a3
                            if AngleGraph.from_string(prim).is_primitive():
                                ret.add(prim)

        return list(ret)

    def to_smarts(self, tag=True):

        if tag:
            return (
                self._atom1.to_smarts(1)
                + self._bond1.to_smarts()
                + self._atom2.to_smarts(2)
                + self._bond1.to_smarts()
                + self._atom1.to_smarts(3)
            )
        else:
            return (
                self._atom1.to_smarts()
                + self._bond1.to_smarts()
                + self._atom2.to_smarts()
                + self._bond1.to_smarts()
                + self._atom1.to_smarts()
            )

    def to_graph(self):
        prims = []
        for field in self._fields:
            obj = getattr(self, field)
            # prims.append(set([x.to_smarts() for x in obj.to_primitives()]))
            prims.append(set(obj.to_primitives()))

        graph = AngleGraph()

        for a1 in prims[0]:
            for bnd in prims[1]:
                for a2 in prims[0]:
                    if a2 < a1:
                        bg = BondGraph(a2, bnd, a1)
                    else:
                        bg = BondGraph(a1, bnd, a2)

                    graph += bg

        return graph


class AngleGraph(ChemGraph):
    def __init__(
        self, atom1=None, bond1=None, atom2=None, bond2=None, atom3=None, sorted=False
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
        self._atom1 = atom1
        self._bond1 = bond1
        self._atom2 = atom2
        self._bond2 = bond2
        self._atom3 = atom3
        self._fields = ["_atom1", "_bond1", "_atom2", "_bond2", "_atom3"]

        if sorted and atom3 < atom1:
            atom1, atom3 = atom3, atom1

        super().__init__()

        self._symmetric = True

    @classmethod
    def from_data(cls, atom1, bond1, atom2, bond2, atom3):
        return cls(atom1, bond1, atom2, bond2, atom3)

    @classmethod
    def from_string_list(cls, string_list, sorted=False):
        atom1 = AtomType.from_string(string_list[0])
        bond1 = BondType.from_string(string_list[1])
        atom2 = AtomType.from_string(string_list[2])
        bond2 = BondType.from_string(string_list[3])
        atom3 = AtomType.from_string(string_list[4])
        if len(string_list) > 5:
            raise NotImplementedError(
                "The SMARTS pattern supplied has not been implemented"
            )
        return cls(atom1, bond1, atom2, bond2, atom3, sorted=sorted)

    @classmethod
    def _split_string(cls, string, sorted=False):
        tokens = cls._smirks_splitter(string, atoms=3)
        return cls.from_string_list(tokens, sorted=sorted)

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

    def to_primitives(self):

        import tqdm

        prims = []

        for field in tqdm.tqdm(
            self._fields, total=len(self._fields), desc="types", disable=True
        ):
            obj = getattr(self, field)
            prims.append(set(obj.to_primitives()))

        ret = set()
        for a1 in tqdm.tqdm(
            prims[0],
            total=len(prims[0]),
            desc="prims1",
            ncols=80,
            disable=np.prod(list(map(len, prims))) < 10000,
        ):
            for bnd1 in prims[1]:
                for a2 in prims[2]:
                    for bnd2 in prims[3]:
                        for a3 in prims[4]:
                            graph = AngleGraph(a1, bnd1, a2, bnd2, a3)
                            if graph.is_primitive() and all([graph != x for x in ret]):
                                ret.add(graph)

        return list(ret)

        prims = []
        for field in self._fields:
            obj = getattr(self, field)
            prims.append(set(obj.to_primitives()))

        ret = set()

        for a1 in prims[0]:
            for bnd1 in prims[1]:
                for a2 in prims[2]:
                    for bnd2 in prims[3]:
                        if bnd2 < bnd1:
                            continue
                        for a3 in prims[4]:
                            if a3 < a1:
                                continue
                            prim = a1 + bnd1 + a2 + bnd2 + a3
                            if AngleGraph.from_string(prim).is_primitive():
                                ret.add(prim)

        return list(ret)

    def is_primitive(self):
        """
        Check the constraints
        """
        if super().is_null():
            return False

        if not super().is_primitive():
            return False

        if not BondGraph(self._atom1, self._bond1, self._atom2).is_primitive():
            return False

        if not BondGraph(self._atom2, self._bond2, self._atom3).is_primitive():
            return False

        return True
        # The central atom must be be connected to more than 1 atom
        if not any(self._atom2._X._v[2:]):
            return False

        if not any(self._atom1._X._v[1:]):
            return False

        if not any(self._atom3._X._v[1:]):
            return False

        # If central atom is connected to 1 H, and both connections are H -> False
        numH = int(self._atom1._symbol._v[1]) + int(self._atom1._symbol._v[1])
        if numH > (len(self._atom2._H._v) - 1):
            return False

        return True

    def to_smarts(self, tag=True, atom_universe=None, bond_universe=None):

        if tag:
            return (
                self._atom1.to_smarts(1, universe=atom_universe)
                + self._bond1.to_smarts(universe=bond_universe)
                + self._atom2.to_smarts(2, universe=atom_universe)
                + self._bond2.to_smarts(universe=bond_universe)
                + self._atom3.to_smarts(3, universe=atom_universe)
            )
        else:
            return (
                self._atom1.to_smarts(universe=atom_universe)
                + self._bond1.to_smarts(universe=bond_universe)
                + self._atom2.to_smarts(universe=atom_universe)
                + self._bond2.to_smarts(universe=bond_universe)
                + self._atom3.to_smarts(universe=atom_universe)
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
        sorted=False,
    ):

        if sorted and atom4 < atom1:
            atom1, bond1, atom2, bond2, atom3, bond3, atom4 = (
                atom4,
                bond3,
                atom3,
                bond2,
                atom2,
                bond1,
                atom1,
            )

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

        super().__init__()

        self._symmetric = True

    # def __contains__(self, o):

    #     if not super().__contains__(o):
    #         rev = o.copy()
    #         rev._fields = rev._fields[::-1]
    #         return super().__contains__(rev)
    #     return True

    @classmethod
    def from_data(cls, atom1, bond1, atom2, bond2, atom3, bond3, atom4):
        return cls(atom1, bond1, atom2, bond2, atom3, bond3, atom4)

    @classmethod
    def from_string_list(cls, string_list, sorted=False):
        atom1 = AtomType.from_string(string_list[0])
        bond1 = BondType.from_string(string_list[1])
        atom2 = AtomType.from_string(string_list[2])
        bond2 = BondType.from_string(string_list[3])
        atom3 = AtomType.from_string(string_list[4])
        bond3 = BondType.from_string(string_list[5])
        atom4 = AtomType.from_string(string_list[6])
        if len(string_list) > 7:
            raise NotImplementedError(
                "The SMARTS pattern supplied has not been implemented"
            )
        return cls(atom1, bond1, atom2, bond2, atom3, bond3, atom4, sorted=sorted)

    @classmethod
    def _split_string(cls, string, sorted=False):
        tokens = cls._smirks_splitter(string, atoms=4)
        return cls.from_string_list(tokens, sorted=sorted)


class DihedralGroup(ChemGraph, abc.ABC):
    def __init__(
        self,
        atom1=None,
        bond1=None,
        atom2=None,
        bond2=None,
        atom3=None,
        bond3=None,
        atom4=None,
        sorted=False,
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
        self._atom1 = atom1 + atom4
        self._bond1 = bond1 + bond3
        self._atom2 = atom2
        self._bond2 = bond2
        self._atom3 = atom3
        self._fields = [
            "_atom1",
            "_bond1",
            "_atom2",
            "_bond2",
            "_atom3",
        ]

        # if atom3 < atom2:
        #     self._fields = [
        #         "_atom1",
        #         "_bond1",
        #         "_atom3",
        #         "_bond2",
        #         "_atom2",
        #     ]

        super().__init__()

    @classmethod
    def from_data(cls, atom1, bond1, atom2, bond2, atom3=None, bond3=None, atom4=None):
        return cls(atom1, bond1, atom2, bond2, atom3, bond3, atom4)

    @classmethod
    def from_string_list(cls, string_list, sorted=False):
        atom1 = AtomType.from_string(string_list[0])
        bond1 = BondType.from_string(string_list[1])
        atom2 = AtomType.from_string(string_list[2])
        bond2 = BondType.from_string(string_list[3])

        atom3, bond3, atom4 = (None, None, None)
        if len(string_list) > 4:
            atom3 = AtomType.from_string(string_list[4])
            bond3 = BondType.from_string(string_list[5])
            atom4 = AtomType.from_string(string_list[6])
        if len(string_list) > 7:
            raise NotImplementedError(
                "The SMARTS pattern supplied has not been implemented"
            )
        return cls(atom1, bond1, atom2, bond2, atom3, bond3, atom4, sorted=False)

    @classmethod
    def _split_string(cls, string, sorted=False):
        tokens = cls._smirks_splitter(string, atoms=4)
        return cls.from_string_list(tokens, sorted=sorted)

    def __contains__(self, o):

        if not super().__contains__(o):
            rev = o.copy()
            rev._fields = [
                "_atom1",
                "_bond1",
                "_atom3",
                "_bond2",
                "_atom2",
            ]
            return super().__contains__(rev)
        return True

    def to_primitives(self):
        prims = []
        for field in self._fields:
            obj = getattr(self, field)
            prims.append(set(obj.to_primitives()))

        ret = set()

        for a1 in prims[0]:
            for bnd1 in prims[1]:
                for a2 in prims[2]:
                    for bnd2 in prims[3]:
                        for a3 in prims[2]:
                            for bnd3 in prims[1]:
                                for a4 in prims[2]:
                                    if a4 < a1:
                                        prim = a1 + bnd1 + a2 + bnd2 + a3 + bnd3 + a4
                                    else:
                                        prim = a4 + bnd3 + a3 + bnd2 + a2 + bnd1 + a1
                                if DihedralGraph.from_string(prim).is_primitive():
                                    ret.add(prim)

        return list(ret)


class TorsionGraph(DihedralGraph):
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

    def is_primitive(self):
        """
        Check the constraints
        """
        if super().is_null():
            return False

        if not super().is_primitive():
            return False

        # If the angles are good, then the dihedral must be good, since it is
        # just two angles that share a central bond
        if not AngleGraph(
            self._atom1, self._bond1, self._atom2, self._bond2, self._atom3
        ).is_primitive():
            return False

        if not AngleGraph(
            self._atom2, self._bond2, self._atom3, self._bond3, self._atom4
        ).is_primitive():
            return False

        return True

    def to_smarts(self, tag=True, atom_universe=None, bond_universe=None):

        if tag:
            return (
                self._atom1.to_smarts(1, universe=atom_universe)
                + self._bond1.to_smarts(universe=bond_universe)
                + self._atom2.to_smarts(2, universe=atom_universe)
                + self._bond2.to_smarts(universe=bond_universe)
                + self._atom3.to_smarts(3, universe=atom_universe)
                + self._bond3.to_smarts(universe=bond_universe)
                + self._atom4.to_smarts(4, universe=atom_universe)
            )
        else:
            return (
                self._atom1.to_smarts(universe=atom_universe)
                + self._bond1.to_smarts(universe=bond_universe)
                + self._atom2.to_smarts(universe=atom_universe)
                + self._bond2.to_smarts(universe=bond_universe)
                + self._atom3.to_smarts(universe=atom_universe)
                + self._bond3.to_smarts(universe=bond_universe)
                + self._atom4.to_smarts(universe=atom_universe)
            )


class TorsionGroup(DihedralGroup):
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

    def __contains__(self, o):

        # Same as DihedralGroup
        return super().__contains__(o)

    def to_smarts(self, tag=True):

        if tag:
            return (
                self._atom1.to_smarts(1)
                + self._bond1.to_smarts()
                + (self._atom2 & self._atom3).to_smarts(2)
                + self._bond2.to_smarts()
                + (self._atom2 & self._atom3).to_smarts(3)
                + self._bond1.to_smarts()
                + self._atom1.to_smarts(4)
            )
        else:
            return (
                self._atom1.to_smarts()
                + self._bond1.to_smarts()
                + (self._atom2 & self._atom3).to_smarts()
                + self._bond2.to_smarts()
                + (self._atom2 & self._atom3).to_smarts()
                + self._bond1.to_smarts()
                + self._atom1.to_smarts()
            )


class OutOfPlaneGraph(DihedralGraph):
    def __init__(
        self,
        atom1=None,
        bond1=None,
        atom2=None,
        bond2=None,
        atom3=None,
        bond3=None,
        atom4=None,
        sorted=False,
    ):
        if sorted:
            if atom1 <= atom3 and atom3 <= atom4:
                atom1, bond1, atom2, bond2, atom3, bond3, atom4 = (
                    atom1,
                    bond1,
                    atom2,
                    bond2,
                    atom3,
                    bond3,
                    atom4,
                )
            elif atom1 <= atom3 and atom4 <= atom3:
                atom1, bond1, atom2, bond2, atom3, bond3, atom4 = (
                    atom1,
                    bond1,
                    atom2,
                    bond3,
                    atom4,
                    bond2,
                    atom3,
                )
            elif atom3 <= atom1 and atom1 <= atom1:
                atom1, bond1, atom2, bond2, atom3, bond3, atom4 = (
                    atom3,
                    bond2,
                    atom2,
                    bond1,
                    atom1,
                    bond3,
                    atom4,
                )
            elif atom3 <= atom4 and atom4 <= atom1:
                atom1, bond1, atom2, bond2, atom3, bond3, atom4 = (
                    atom3,
                    bond2,
                    atom2,
                    bond3,
                    atom4,
                    bond1,
                    atom1,
                )
            elif atom4 <= atom1 and atom1 <= atom3:
                atom1, bond1, atom2, bond2, atom3, bond3, atom4 = (
                    atom4,
                    bond3,
                    atom2,
                    bond1,
                    atom1,
                    bond2,
                    atom3,
                )
            else:
                atom1, bond1, atom2, bond2, atom3, bond3, atom4 = (
                    atom4,
                    bond3,
                    atom2,
                    bond2,
                    atom3,
                    bond1,
                    atom1,
                )

        super().__init__(
            atom1=atom1,
            bond1=bond1,
            atom2=atom2,
            bond2=bond2,
            atom3=atom3,
            bond3=bond3,
            atom4=atom4,
        )
        self._symmetric = False

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

    def is_primitive(self):
        """
        Check the constraints
        """
        if super().is_null():
            return False

        if not super().is_primitive():
            return False

        # If the angles are good, then the dihedral must be good, since it is
        # just two angles that share a central bond
        if not AngleGraph(
            self._atom1, self._bond1, self._atom2, self._bond2, self._atom3
        ).is_primitive():
            return False

        if not AngleGraph(
            self._atom1, self._bond1, self._atom2, self._bond3, self._atom4
        ).is_primitive():
            return False

        return True

    def _permutations(self):
        """"""

        return (
            (0, 1, 2, 3, 4, 5, 6),
            (0, 1, 2, 5, 6, 3, 4),
            (4, 3, 2, 1, 0, 5, 6),
            (4, 3, 2, 5, 6, 1, 0),
            (6, 5, 2, 1, 0, 3, 4),
            (6, 5, 2, 3, 4, 1, 0),
        )

    def __contains__(self, o):

        for order in self._permutations():
            rev = o.copy()
            rev._fields = [rev._fields[i] for i in order]
            if super().__contains__(o):
                return True
        return False
        # if not super().__contains__(o):
        #     rev = o.copy()
        #     rev._fields = [
        #         "_atom1",
        #         "_bond1",
        #         "_atom2",
        #         "_bond2",
        #         "_atom4",
        #         "_bond3",
        #         "_atom3",
        #     ]
        # else:
        #     return True
        # if not super().__contains__(rev):
        #     rev = o.copy()
        #     rev._fields = [
        #         "_atom3",
        #         "_bond2",
        #         "_atom2",
        #         "_bond1",
        #         "_atom1",
        #         "_bond3",
        #         "_atom4",
        #     ]
        # else:
        #     return True
        # if not super().__contains__(rev):
        #     rev = o.copy()
        #     rev._fields = [
        #         "_atom3",
        #         "_bond2",
        #         "_atom2",
        #         "_bond3",
        #         "_atom4",
        #         "_bond1",
        #         "_atom1",
        #     ]
        # else:
        #     return True
        # if not super().__contains__(rev):
        #     rev = o.copy()
        #     rev._fields = [
        #         "_atom4",
        #         "_bond3",
        #         "_atom2",
        #         "_bond3",
        #         "_atom3",
        #         "_bond1",
        #         "_atom1",
        #     ]
        # else:
        #     return True
        # if not super().__contains__(rev):
        #     rev = o.copy()
        #     rev._fields = [
        #         "_atom4",
        #         "_bond3",
        #         "_atom2",
        #         "_bond1",
        #         "_atom1",
        #         "_bond3",
        #         "_atom3",
        #     ]
        # else:
        #     return True
        # return super().__contains__(rev)

    def to_smarts(self, tag=True, atom_universe=None, bond_universe=None):

        if tag:
            return (
                self._atom1.to_smarts(1, universe=atom_universe)
                + self._bond1.to_smarts(universe=bond_universe)
                + self._atom2.to_smarts(2, universe=atom_universe)
                + "("
                + self._bond2.to_smarts(universe=bond_universe)
                + self._atom3.to_smarts(3, universe=atom_universe)
                + ")"
                + self._bond3.to_smarts(universe=bond_universe)
                + self._atom4.to_smarts(4, universe=atom_universe)
            )
        else:
            return (
                self._atom1.to_smarts(universe=atom_universe)
                + self._bond1.to_smarts(universe=bond_universe)
                + self._atom2.to_smarts(universe=atom_universe)
                + "("
                + self._bond2.to_smarts(universe=bond_universe)
                + self._atom3.to_smarts(universe=atom_universe)
                + ")"
                + self._bond3.to_smarts(universe=bond_universe)
                + self._atom4.to_smarts(universe=atom_universe)
            )


class OutOfPlaneGroup(DihedralGroup):
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

    def __contains__(self, o):

        if not super().__contains__(o):
            rev = o.copy()
            rev._fields = [
                "_atom3",
                "_bond2",
                "_atom2",
                "_bond1",
                "_atom1",
            ]
            return super().__contains__(rev)
        return True

    def to_smarts(self, tag=True):

        if tag:
            return (
                self._atom1.to_smarts(1)
                + self._bond1.to_smarts()
                + self._atom2.to_smarts(2)
                + "("
                + self._bond2.to_smarts()
                + self._atom3.to_smarts(3)
                + ")"
                + self._bond1.to_smarts()
                + self._atom1.to_smarts(4)
            )
        else:
            return (
                self._atom1.to_smarts()
                + self._bond1.to_smarts()
                + self._atom2.to_smarts()
                + "("
                + self._bond2.to_smarts()
                + self._atom3.to_smarts()
                + ")"
                + self._bond1.to_smarts()
                + self._atom1.to_smarts()
            )


ATOM_UNIVERSE = AtomType()
for i in [1, 6, 8]:
    ATOM_UNIVERSE._symbol[i] = True
# ATOM_UNIVERSE._symbol[8] = False
ATOM_UNIVERSE._H[:5] = True
# ATOM_UNIVERSE._H[3] = False
# ATOM_UNIVERSE._H[0] = False
# ATOM_UNIVERSE._H[1] = False
# ATOM_UNIVERSE._H[2] = False
# ATOM_UNIVERSE._H[4] = False

ATOM_UNIVERSE._X[:5] = True
# ATOM_UNIVERSE._X[3] = False
# ATOM_UNIVERSE._X[2] = False

ATOM_UNIVERSE._x[:4] = True

ATOM_UNIVERSE._r[:6] = True
ATOM_UNIVERSE._aA[:2] = True

BOND_UNIVERSE = BondType()
BOND_UNIVERSE._order[1:5] = True
BOND_UNIVERSE._aA[:2] = True
