import openforcefield.typing.engines.smirnoff
import openforcefield.typing.engines.smirnoff.parameters

# currently not possible to wrap handlers since the TK does this thing
# where it searches all subclasses, and then fails when the subclass tries
# to use the same PH name. So, we cannot subclass any handlers unless they
# provide a different tag
from openforcefield.typing.engines.smirnoff.parameters import (
    AngleHandler, BondHandler, ImproperTorsionHandler, ProperTorsionHandler, vdWHandler)


class ValenceDict(openforcefield.typing.engines.smirnoff.parameters.ValenceDict):
    pass


class ImproperDict(openforcefield.typing.engines.smirnoff.parameters.ImproperDict):
    pass



# class AngleHandler(openforcefield.typing.engines.smirnoff.parameters.AngleHandler):
#     pass


# class BondHandler(openforcefield.typing.engines.smirnoff.parameters.BondHandler):
#     pass


# class ImproperTorsionHandler(
#     openforcefield.typing.engines.smirnoff.parameters.ImproperTorsionHandler
# ):
#     pass


# class ProperTorsionHandler(
#     openforcefield.typing.engines.smirnoff.parameters.ProperTorsionHandler
# ):
#     pass


class ParameterList(openforcefield.typing.engines.smirnoff.parameters.ParameterList):
    pass


# class vdWHandler(openforcefield.typing.engines.smirnoff.parameters.vdWHandler):
#     pass


class ForceField(openforcefield.typing.engines.smirnoff.ForceField):
    pass
