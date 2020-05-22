import treedi.tree
import numpy as np

class AngleOperation( treedi.tree.TreeOperation):

    def __init__( self, source, name):
        super().__init__( source, name)

    @staticmethod
    def measure(mol, idx):
        """calculates angle between origin and consecutive atom pairs"""
        atoms = mol.get( "geometry")[ np.newaxis, :, :]
        mags = np.linalg.norm(atoms[:,[idx[0],idx[2]],:] - atoms[:,idx[1],:][:,np.newaxis,:], axis=2)
        atoms_trans = atoms - atoms[:,idx[1],:][:,np.newaxis,:]
        unit = atoms_trans[:,[idx[0],idx[2]],:] / mags[:,:,np.newaxis]
        costheta = (unit[:,0,:] * unit[:,1,:]).sum(axis=1)
        np.clip(costheta, -1.0, 1.0, out=costheta)
        ret = np.arccos(costheta)*180/np.pi
        return ret

    def op(self, mol, idx):
        return self.measure(mol, idx)

class BondOperation( treedi.tree.TreeOperation):

    def __init__( self, source, name):
        super().__init__( source, name)

    @staticmethod
    def measure(mol, idx):
        """calculates distance from first atom to remaining atoms"""
        atoms = mol.get( "geometry")[ np.newaxis, :, :]
        #print( "Have", len(atoms[0]), "atoms and index is", idx)
        return np.linalg.norm(atoms[:,idx[1],:] - atoms[:,idx[0],:], axis=1)

    def op(self, mol, idx):
        return self.measure(mol, idx)

class TorsionOperation( treedi.tree.TreeOperation):

    def __init__( self, source, name):
        super().__init__( source, name)
    
    @staticmethod
    def measure( mol, idx):
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
        
        p = mol.get( "geometry")[idx]
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
        
        p = mol.get( "geometry")[np.newaxis, idx, :]
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

class ImproperTorsionOperation( treedi.tree.TreeOperation):

    def __init__( self, source, name):
        super().__init__( source, name)

    @staticmethod
    def measure(mol, idx):
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

    def op(self, mol, idx):
        return self.measure(mol, idx)
