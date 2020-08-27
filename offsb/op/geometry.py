import treedi.tree
import numpy as np
from abc import (ABC, abstractmethod)
import logging
from offsb.tools.util import flatten_list
import offsb.rdutil.mol

class GeometryOperation(treedi.tree.TreeOperation, ABC):

    def __init__(self, source, name):
        super().__init__(source, name)
        self._select="Molecule"
        self.processes=1

    def _unpack_result(self, ret):
        self.db.update(ret)

    def apply(self, targets=None):
        super().apply(self._select, targets=targets)

    def _generate_apply_kwargs(self, i, target, kwargs={}):
        entry = self.source.source.node_iter_to_root(target, select="Entry")
        entry = next(entry)
        entry_node = entry
        entry_name = str(entry)

        self.logger.debug("{}".format(entry_node))
        entry = self.source.source.db[entry.payload]['data']
        out_str = ""
        CIEHMS = "canonical_isomeric_explicit_hydrogen_mapped_smiles"
        smi = entry.attributes[CIEHMS]
        if "initial_molecule" in entry.dict():
            qcid = entry.dict()["initial_molecule"]
        elif "initial_molecules" in entry.dict():
            qcid = entry.dict()["initial_molecules"]
        else:
            out_str += "{:d} initial mol was empty: {:s}".format(i, str(qcid))
            return {"error": out_str}

        if isinstance(qcid, set):
            qcid = list(qcid)
        if isinstance(qcid, list):
            qcid = str(qcid[0])

        qcmolid = "QCM-" + qcid

        if qcmolid not in self.source.source.db:
            out_str += "{:d} initial mol was empty: {:s}".format(i, str(qcmolid))
            return {"error": out_str}

        if "data" in self.source.source.db.get(qcmolid):
            qcmol = self.source.source.db.get(qcmolid).get("data")
        else:
            out_str += "{:d} initial mol was empty: {:s}".format(i, str(qcmolid))
            return {"error": out_str}

        mol = offsb.rdutil.mol.rdmol_from_smiles_and_qcmol(smi, qcmol)

        map_idx = {a.GetIdx(): a.GetAtomMapNum() for a in mol.GetAtoms()}

        if any([x == 0 for x in map_idx.values()]) and \
           any([x != 0 for x in map_idx.values()]):
            raise IndexError("Partial map index read; refuse to proceed")
        
        mol = self.source.source.db[target.payload]['data']['geometry']

        obj = self.source.db[entry_node.payload]
        masks = flatten_list([x for x in obj["data"].values()], times=1)
        self.logger.debug("Have masks {}".format(masks))



        return {"masks": masks, "map_idx": map_idx, "mol": mol, "name": self.name, "op": self.op,
                "entry": entry_name}

    @staticmethod
    def apply_single(i, target, kwargs=None):

        if kwargs is None:
            raise Exception("Geometry operation not given necessary config")
        # entry = self.source.source.node_iter_to_root(target, select="Entry")
        # entry = next(entry)

        # mol = self.source.source.db[target.payload]

        # obj = self.source.db[self.source[entry.index].payload]
        # masks = obj["data"]

        masks = kwargs['masks']
        mol = kwargs['mol']
        op = kwargs['op']
        entry = kwargs['entry']
        map_idx = kwargs['map_idx']

        ret = {}
        calcs = 0
        out_str = ""
        debug_str = ""

        # need to map to chemical indices e.g. as the appear in cmiles
        # and the symbols/geometry are based on the map
        # for i in range(len(masks)):
        #     masks[i] = tuple([map_idx[j] - 1 for j in masks[i]])

        # breakpoint()
        for mask in masks:

            if map_idx is not None:
                mask_mapped = tuple([map_idx[i]-1 for i in mask])
            # mask = [i-1 for i in mask]
            debug_str += "Measuring {} on entry {} molecule {}\n".format(
                str(mask), entry, target.payload)
            vals = None
            try:
                vals = op(mol, mask_mapped)
                calcs += 1
            except Exception as e:
                out_str += "Calculation failed for" + str(target) + "\n"
                out_str += "Error:\n" + str(e) + '\n'

            ret[tuple(mask)] = vals
            debug_str += "value for {}: {}\n".format(str(mask), vals)

        # out_str = "calculated: {}\n".format(calcs)
        return {target.payload: out_str, "debug": debug_str, "return": {target.payload: ret}}

    # def apply(self, targets=None, select="Molecule"):
    #     calcs = 0
    #     self.source.apply(targets=targets)
    #     if targets is None:
    #         entries = list(self.source.source.iter_entry())
    #     else:
    #         entries = targets
    #     if not hasattr(entries, "__iter__"):
    #         entries = [entries]

    #     for entry in entries:
    #         mol_calcs = 0
    #         obj = self.source.db[self.source[entry.index].payload]

    #         masks = obj["data"]

    #         for mol_node in self.node_iter_depth_first(entry, select=select):
    #             mol = self.source.source.db[mol_node.payload]
    #             for mask in masks:
    #                 ret = {}
    #                 if mol_node.payload in self.db:
    #                     ret = self.db.get( mol_node.payload)
    #                 else:
    #                     self.db[mol_node.payload] = dict()

    #                 ret[tuple(mask)] = self.op( mol["data"], [i-1 for i in mask])
    #                 self.db[mol_node.payload].update( ret)
    #                 mol_calcs += 1

    #         calcs += mol_calcs
    #     print(self.name + " calculated: {}".format( calcs))

class AngleOperation(GeometryOperation):

    def __init__( self, source, name):
        super().__init__( source, name)

    @staticmethod
    def measure(mol, idx):
        """calculates angle between origin and consecutive atom pairs"""
        atoms = mol[ np.newaxis, :, :]
        mags = np.linalg.norm(atoms[:,[idx[0],idx[2]],:] - atoms[:,idx[1],:][:,np.newaxis,:], axis=2)
        atoms_trans = atoms - atoms[:,idx[1],:][:,np.newaxis,:]
        unit = atoms_trans[:,[idx[0],idx[2]],:] / mags[:,:,np.newaxis]
        costheta = (unit[:,0,:] * unit[:,1,:]).sum(axis=1)
        np.clip(costheta, -1.0, 1.0, out=costheta)
        ret = np.arccos(costheta)*180/np.pi
        return ret

    def op(self, mol, idx):
        return self.measure(mol, idx)

class BondOperation(GeometryOperation):

    def __init__( self, source, name):
        super().__init__( source, name)

    @staticmethod
    def measure(mol, idx):
        """calculates distance from first atom to remaining atoms"""
        atoms = mol[ np.newaxis, :, :]
        #print( "Have", len(atoms[0]), "atoms and index is", idx)
        return np.linalg.norm(atoms[:,idx[1],:] - atoms[:,idx[0],:], axis=1)

    def op(self, mol, idx):
        return self.measure(mol, idx)

class TorsionOperation(GeometryOperation):

    def __init__( self, source, name):
        super().__init__( source, name)
    
    @staticmethod
    def measure( mol, idx):
        """calculates proper torsion of [i, j, k, l]"""
        atoms = mol[np.newaxis, :, :]
        noncenter = [idx[0]]+idx[2:4]
        mags = np.linalg.norm(atoms[:,noncenter,:] - atoms[:,idx[1],:][:,np.newaxis,:], axis=2)
        atoms_trans = atoms - atoms[:,idx[1],:][:,np.newaxis,:]
        unit = atoms_trans[:,noncenter,:] / mags[:,:,np.newaxis]
        #these are all Nx3
        v0 = -unit[:,0,:]
        v1 = unit[:,1,:]
        v2 = unit[:,2,:]-unit[:,1,:]

        w1 = np.cross(v0,v1)
        w2 = np.cross(v1,v2)

        w1_mag = np.linalg.norm(w1,axis=1)
        w2_mag = np.linalg.norm(w2,axis=1)

        mask = (w1_mag * w2_mag) > 0
        # should be Nx1 costhetas
        costheta = np.ones((atoms.shape[0]))
        costheta[mask]= (w1[mask] * w2[mask]).sum(axis=1) / (w1_mag[mask]*w2_mag[mask])
        np.clip(costheta, -1.0, 1.0, out=costheta)
        sgn = np.sign(costheta)

        theta = np.arccos(costheta)*180/np.pi

        #distance = np.zeros((atoms.shape[0]))
        #distance[mask] = ((w2[mask]*v0[mask]).sum(axis=1)/w2_mag[mask])
        #theta[distance > 0] = 180 - theta[distance > 0]
        #theta[np.abs(theta) > 180.0] %= 180.0
        return theta

    @staticmethod
    def measure_praxeolitic_single(mol, idx):
        """Praxeolitic formula
        1 sqrt, 1 cross product
        
        From:
        https://stackoverflow.com/questions/20305272/


        TODO: Currently doesn' work, since I am reworking it to handle entire
        trajectories (MxNxD)
        """
        
        p = mol[idx]
        p0 = p[0]
        p1 = p[1]
        p2 = p[2]
        p3 = p[3]

        b0 = -1.0*(p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2

        # normalize b1 so that it does not influence magnitude of vector
        # rejections that come next
        b1 /= np.linalg.norm(b1)

        # vector rejections
        # v = projection of b0 onto plane perpendicular to b1
        #   = b0 minus component that aligns with b1
        # w = projection of b2 onto plane perpendicular to b1
        #   = b2 minus component that aligns with b1
        v = b0 - np.dot(b0, b1)*b1
        w = b2 - np.dot(b2, b1)*b1

        # angle between v and w in a plane is the torsion angle
        # v and w may not be normalized but that's fine since tan is y/x
        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)

        return np.degrees(np.arctan2(y, x))

    @staticmethod
    def measure_praxeolitic(mol, idx):
        """Praxeolitic formula
        1 sqrt, 1 cross product
        
        From:
        https://stackoverflow.com/questions/20305272/


        TODO: Currently doesn' work, since I am reworking it to handle entire
        trajectories (MxNxD)
        """
        
        p = mol[np.newaxis, idx, :]
        p0 = p[:,0]
        p1 = p[:,1]
        p2 = p[:,2]
        p3 = p[:,3]

        b0 = -1.0*(p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2

        # normalize b1 so that it does not influence magnitude of vector
        # rejections that come next
        b1 /= np.linalg.norm(b1, axis=1)

        # vector rejections
        # v = projection of b0 onto plane perpendicular to b1
        #   = b0 minus component that aligns with b1
        # w = projection of b2 onto plane perpendicular to b1
        #   = b2 minus component that aligns with b1
        #v = b0 - np.dot(b0, b1)*b1
        #w = b2 - np.dot(b2, b1)*b1
        v = b0 - (b0 * b1).sum(axis=0)*b1
        w = b2 - (b2 * b1).sum(axis=0)*b1

        # angle between v and w in a plane is the torsion angle
        # v and w may not be normalized but that's fine since tan is y/x
        #x = np.dot(v, w)
        #y = np.dot(np.cross(b1, v), w)

        x = (v * w).sum(axis=0)
        y = np.atleast_2d((np.cross(b1, v) * w).sum(axis=0))

        ret = np.degrees(np.arctan2(y, x))
        return ret

    def op(self, mol, idx):
        return self.measure_praxeolitic(mol, idx)

class ImproperTorsionOperation(GeometryOperation):

    def __init__( self, source, name):
        super().__init__(source, name)

    @staticmethod
    def measure(mol, idx):
        """calculates improper torsion of [i, center, j, k]"""
        atoms = mol[np.newaxis, :, :]
        noncenter = [idx[0]]+idx[2:4]
        mags = np.linalg.norm(atoms[:,noncenter,:] - atoms[:,idx[1],:][:,np.newaxis,:], axis=2)
        atoms_trans = atoms - atoms[:,idx[1],:][:,np.newaxis,:]
        unit = atoms_trans[:,noncenter,:] / mags[:,:,np.newaxis]
        #these are all Nx3
        v0 = -unit[:,0,:]
        v1 = unit[:,1,:]-unit[:,0,:]
        v2 = unit[:,1,:]-unit[:,2,:]

        w1 = np.cross(v0,v1)
        w2 = np.cross(v1,v2)

        w1_mag = np.linalg.norm(w1,axis=1)
        w2_mag = np.linalg.norm(w2,axis=1)

        mask = (w1_mag * w2_mag) > 0
        # should be Nx1 costhetas
        costheta = np.ones((atoms.shape[0]))
        costheta[mask]= (w1[mask] * w2[mask]).sum(axis=1) / (w1_mag[mask]*w2_mag[mask])
        np.clip(costheta, -1.0, 1.0, out=costheta)

        theta = np.arccos(costheta)*180/np.pi

        distance = np.zeros((atoms.shape[0]))
        distance[mask] = ((w2[mask]*v0[mask]).sum(axis=1)/w2_mag[mask])
        #theta[distance > 0] = 180 - theta[distance > 0]
        theta[distance < 0] *= -1
            
        return theta

    def op(self, mol, idx):
        return self.measure(mol, idx)
