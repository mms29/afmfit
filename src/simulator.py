import matplotlib.pyplot as plt
import numpy as np
from numba import njit
import time
from src.utils import get_sphere_near, get_sphere, rotation_matrix_from_vectors
from . import AFMIZE_PATH
import os

class AFMSimulator:

    def __init__(self, size, vsize, beta, sigma, cutoff):
        self.size=size
        self.vsize=vsize
        self.beta=beta
        self.sigma=sigma
        self.cutoff=cutoff
        self.n_pix_cutoff = int(np.ceil(cutoff / vsize) * 2 + 1)

    def pdb2afm(self, pdb, zshift=None):
        r = pdb.get_radius()
        pix = _select_pixels(pdb.coords, size=self.size, voxel_size=self.vsize, n_pix_cutoff=self.n_pix_cutoff)
        if zshift is None:
            zshift = -pdb.coords[2].min()
        afm = _pdb2afm(coords=pdb.coords, size=self.size, vsize=self.vsize, pix=pix, beta=self.beta, r=r ,
                 sigma=self.sigma, n_pix_cutoff=self.n_pix_cutoff, zshift=zshift)
        return afm

    def pdb2afm_grad(self, pdb,nma, psim, zshift):
        r = pdb.get_radius()
        pix = _select_pixels(pdb.coords, size=self.size, voxel_size=self.vsize, n_pix_cutoff=self.n_pix_cutoff)

        grad = _pdb2afm_grad_njit(coords=pdb.coords, modes=nma.linear_modes, psim=psim, pix=pix, size=self.size,
            zshift=zshift, vsize=self.vsize, beta=self.beta, r=r, sigma=self.sigma, n_pix_cutoff=self.n_pix_cutoff)
        return grad

    def get_projection_library(self, pdb,  zshift_range=None, angular_dist=10,
                           verbose=True, near_point=None, near_cutoff=30, plot=False):
        ZERO_CUTOFF =5.0 # Angstrom

        # Compute directions views
        if near_point is not None:  points = get_sphere_near(angular_dist, near_point, near_cutoff)
        else:                       points = get_sphere(angular_dist)

        # Initiate arrays
        num_pts = len(points)
        rotations = np.zeros((num_pts, 3, 3))
        z_shifts = np.zeros((num_pts))
        if zshift_range is None:
            Z_SHIFT_SEARCH = 0
            image_library = np.zeros((num_pts,  self.size, self.size))
        else:
            Z_SHIFT_SEARCH = len(zshift_range)
            image_library = np.zeros((num_pts,Z_SHIFT_SEARCH,  self.size, self.size))

        # Loop over all viewing directions
        dtime = time.time()
        for i in range(num_pts):
            if verbose:
                if int((i + 1) % (num_pts / 10)) == 0:
                    print("%i %%" % int(100 * (i + 1) / num_pts))

            # get rotation matrix
            rotations[i] = rotation_matrix_from_vectors(points[i])

            # Rotate coordinates
            rot_pdb = pdb.copy()
            rot_pdb.coords = np.dot(rotations[i], pdb.coords.T).T
            z_shifts[i] = -rot_pdb.coords[:, 2].min()

            # Simulate images
            img = self.pdb2afm(rot_pdb, zshift = z_shifts[i])

            # Simulate images in Z direction
            if Z_SHIFT_SEARCH>0:
                for z in range(Z_SHIFT_SEARCH):
                    img_zshift = img.copy()
                    img_zshift[img_zshift>=ZERO_CUTOFF]+= zshift_range[z]
                    img_zshift[img_zshift<0.0] = 0.0
                    image_library[i, z] = img_zshift
            else:
                image_library[i] = img

        if verbose:
            print("\t TIME_TOTAL => 100 %% ;  %.4f s" % (time.time() - dtime))

        return ImageLibrary(image_library, rotations, points, z_shifts, zshift_range)

    def afmize(self, pdb, probe_r, probe_a, noise, prefix):
        pdb.write_pdb(prefix + ".pdb")
        pdbfn = prefix + ".pdb"

        with open(prefix + ".toml", "w") as f:
            f.write("file.input           = \"%s\" \n" % pdbfn)
            f.write("file.output.basename = \"%s\" \n" % prefix)
            f.write("file.output.formats  = [\"tsv\", \"svg\"] \n")
            f.write("probe.size           = {radius = \"%.1fangstrom\", angle = %.1f} \n" % (probe_r, probe_a))
            f.write("resolution.x         = \"%.3fangstrom\" \n" % self.vsize)
            f.write("resolution.y         = \"%.3fangstrom\" \n" % self.vsize)
            f.write("resolution.z         = \"0.1angstrom\" \n")
            f.write("range.x              = [\"%.3fangstrom\", \"%.3fangstrom\"] \n" % (
            -self.size * self.vsize / 2, self.size * self.vsize / 2))
            f.write("range.y              = [\"%.3fangstrom\", \"%.3fangstrom\"] \n" % (
            -self.size * self.vsize / 2, self.size * self.vsize / 2))
            f.write("scale_bar.length     = \"5.0nm\" \n")
            f.write("stage.align          = true \n")
            f.write("stage.position       = 0.0 \n")
            f.write("noise                = \"%.1fangstrom\" \n" % noise)

        os.system("%s %s.toml" % (AFMIZE_PATH, prefix))

        return np.loadtxt(prefix + ".tsv").T

class ImageLibrary:

    def __init__(self, imgs, rotations, directions, zshifts, zshift_range=None):
        self.imgs = imgs
        self.nimg = len(imgs)
        self.rotations = rotations
        self.directions = directions
        self.zshifts = zshifts
        self.zshift_range = zshift_range
        if len(imgs.shape) == 3:
            self.n_zshift =0
        else:
            self.n_zshift = imgs.shape[1]

    def show(self):
        plt.ion()
        fig, ax = plt.subplots(1, 1)
        for i in range(self.imgs.shape[0]):
            if self.n_zshift > 0 :
                for z in range(self.n_zshift):
                    ax.clear()
                    ax.imshow(self.imgs[i, z].T, cmap="afmhot", origin="lower")
                    fig.canvas.draw()
                    fig.canvas.flush_events()
            else:
                ax.clear()
                ax.imshow(self.imgs[i].T, cmap="afmhot", origin="lower")
                fig.canvas.draw()
                fig.canvas.flush_events()




@njit
def _pdb2afm_grad_njit(coords, modes, psim, pix, size, vsize, beta, r, sigma, n_pix_cutoff, zshift):
    dq = np.zeros((modes.shape[0], size,size))

    n_atoms = coords.shape[0]
    sgmam2 = 1/ (2 * (sigma ** 2))
    bsgm2 = beta/sigma**2
    for i in range(n_atoms):
        for x in range(pix[i,0], pix[i,0] + n_pix_cutoff):
            for y in range(pix[i,1], pix[i,1] + n_pix_cutoff):
                mu = (np.array([x,y]) - size / 2) * vsize

                expnt = np.exp(
                    -np.square(np.linalg.norm(coords[i,:2] - mu)) *sgmam2
                    +(coords[i,2]+r[i] + zshift)/beta
                    - psim[x,y]/beta
                )
                dpsim = np.array([expnt, expnt, expnt])
                dpsim[:2] *=  (coords[i,:2] - mu) * -(bsgm2)
                dq[:,x,y] +=  np.dot(modes[:,i], dpsim)

    return dq



@njit
def _pdb2afm(coords, size, vsize, pix, beta, r , sigma, n_pix_cutoff, zshift):
    n_atoms = coords.shape[0]
    img = np.zeros((size,size))
    sgmam2 = 1/ (2 * (sigma ** 2))
    for i in range(n_atoms):
        for x in range(pix[i,0], pix[i,0] + n_pix_cutoff):
            for y in range(pix[i,1], pix[i,1] + n_pix_cutoff):
                mu = (np.array([x,y]) - size / 2) * vsize
                expnt = np.exp(
                    -np.square(np.linalg.norm(coords[i,:2] - mu)) *sgmam2
                    +(coords[i,2] + zshift+r[i])/beta
                )
                img[x,y] += expnt
    return beta * np.log(1+ img)

@njit
def _select_pixels(coord, size, voxel_size, n_pix_cutoff):
    n_atoms = coord.shape[0]
    threshold = (n_pix_cutoff-1)//2
    l=np.zeros((n_atoms,2), dtype=np.int32)
    for i in range(n_atoms):
        l[i,0] = np.floor(coord[i,0]/voxel_size -threshold + size/2)
        l[i,1] = np.floor(coord[i,1]/voxel_size -threshold + size/2)
    if (np.max(l) >= size or np.min(l)<0) or (np.max(l+n_pix_cutoff) >= size or np.min(l+n_pix_cutoff)<0):
        raise RuntimeError("ERROR : Atomic coordinates got outside the box")
    return l

