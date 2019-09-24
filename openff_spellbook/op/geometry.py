import treedi.tree
import numpy as np

class AngleOperation( treedi.tree.TreeOperation):

    def __init__( self, source, name):
        super().__init__( source, name)

    def op(self, mol, idx):
        """calculates angle between origin and consecutive atom pairs"""
        atoms = mol.get( "geometry")[ np.newaxis, :, :]
        mags = np.linalg.norm(atoms[:,[idx[0],idx[2]],:] - atoms[:,idx[1],:][:,np.newaxis,:], axis=2)
        atoms_trans = atoms - atoms[:,idx[1],:][:,np.newaxis,:]
        unit = atoms_trans[:,[idx[0],idx[2]],:] / mags[:,:,np.newaxis]
        costheta = (unit[:,0,:] * unit[:,1,:]).sum(axis=1)
        np.clip(costheta, -1.0, 1.0, out=costheta)
        ret = np.arccos(costheta)*180/np.pi
        return ret

class BondOperation( treedi.tree.TreeOperation):

    def __init__( self, source, name):
        super().__init__( source, name)

    def op(self, mol, idx):
        """calculates distance from first atom to remaining atoms"""
        atoms = mol.get( "geometry")[ np.newaxis, :, :]
        #print( "Have", len(atoms[0]), "atoms and index is", idx)
        return np.linalg.norm(atoms[:,idx[1],:] - atoms[:,idx[0],:], axis=1)

class TorsionOperation( treedi.tree.TreeOperation):

    def __init__( self, source, name):
        super().__init__( source, name)

    def op(self, mol, idx):
        """calculates proper torsion of [i, j, k, l]"""
        atoms = mol.get( "geometry")[ np.newaxis, :, :]
        noncenter = [idx[0]]+idx[2:]
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

        theta = np.arccos(costheta)*180/np.pi

        #distance = np.zeros((atoms.shape[0]))
        #distance[mask] = ((w2[mask]*v0[mask]).sum(axis=1)/w2_mag[mask])
        ##theta[distance > 0] = 180 - theta[distance > 0]
        theta[np.abs(theta) >= 180] %= 180.0
        return theta

class ImproperTorsionOperation( treedi.tree.TreeOperation):

    def __init__( self, source, name):
        super().__init__( source, name)

    def op(self, mol, idx):
        """calculates improper torsion of [i, center, j, k]"""
        atoms = mol.get( "geometry")[ np.newaxis, :, :]
        noncenter = [idx[0]]+idx[2:]
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
