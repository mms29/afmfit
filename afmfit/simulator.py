#     AFMfit - Fitting package for Atomic Force Microscopy data
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
from afmfit.utils import get_sphere_full
from afmfit.image import ImageLibrary

import numpy as np
from numba import njit
import os
import tqdm
from multiprocessing.sharedctypes import RawArray
from numpy import frombuffer
from multiprocessing import Pool

class AFMSimulator:

    def __init__(self, size, vsize, beta, sigma, cutoff, ):
        """

        :param size:
        :param vsize:
        :param beta:
        :param sigma:
        :param cutoff:
        """
        self.size=size
        self.vsize=vsize
        self.beta=beta
        self.sigma=sigma
        self.cutoff=cutoff
        self.n_pix_cutoff = int(np.ceil(cutoff / vsize) * 2 + 1)

    def pdb2afm(self, pdb, zshift=None):
        r = pdb.get_radius()[pdb.active_atoms]
        coords = pdb.coords[pdb.active_atoms]
        pix = _select_pixels(coords, size=self.size, voxel_size=self.vsize, n_pix_cutoff=self.n_pix_cutoff)
        if zshift is None:
            zshift = -pdb.coords[:, 2].min()
        afm = _pdb2afm_njit(coords=coords, size=self.size, vsize=self.vsize, pix=pix, beta=self.beta, r=r ,
                 sigma=self.sigma, zshift=zshift)
        return afm

    def pdb2afm_grad(self, pdb,nma, psim, zshift):
        r = pdb.get_radius()[pdb.active_atoms]
        coords = pdb.coords[pdb.active_atoms]
        linear_modes = nma.linear_modes[:,pdb.active_atoms]
        pix = _select_pixels(pdb.coords, size=self.size, voxel_size=self.vsize, n_pix_cutoff=self.n_pix_cutoff)
        grad = _pdb2afm_grad_njit(coords=coords, modes=linear_modes, psim=psim, pix=pix, size=self.size,
            zshift=zshift, vsize=self.vsize, beta=self.beta, r=r, sigma=self.sigma)
        return grad

    def project_library(self, n_cpu, pdb, angular_dist, init_zshift = None, verbose=True, zshift_range=None,
                        near_angle=None, near_angle_cutoff = None, true_zshift=True):
        ZERO_CUTOFF = 5.0  # Angstrom

        # Compute directions views
        angles = get_sphere_full(angular_dist, near_angle=near_angle, near_angle_cutoff=near_angle_cutoff)

        #inputs
        n_angles, _ = angles.shape
        n_zshifts = len(zshift_range)
        n_imgs = n_angles*n_zshifts

        # Allocate Arrays
        imageLibrarySharedArray = RawArray("f", n_imgs*self.size*self.size)
        zshiftRawArray = RawArray("d", n_imgs)
        angles_z = np.zeros((n_imgs,3))
        workdata=[]

        for i in range(n_angles):
            workdata.append([i , angles[i]])

        # Run multiprocess
        p = Pool(n_cpu, initializer=self.init_projLibrary_processes, initargs=(imageLibrarySharedArray, zshiftRawArray, n_imgs,
                n_zshifts, ZERO_CUTOFF, zshift_range, self, pdb, init_zshift, true_zshift))
        for _ in tqdm.tqdm(p.imap_unordered(self.run_projLibrary_process, workdata), total=len(workdata), desc="Project Library"
                           , disable=not verbose):
            pass

        del workdata

        for i in range(n_angles):
            for z in range(n_zshifts):
                angles_z[i*n_zshifts + z] = angles[i]
        z_shifts = frombuffer(zshiftRawArray, dtype=np.float64, count=len(zshiftRawArray))

        ngroup = n_zshifts * 360 // angular_dist
        nviews = n_imgs // ngroup
        view_group = np.zeros((nviews, ngroup), dtype=int)
        arr = np.arange(ngroup)
        for i in range(nviews):
            view_group[i] = arr + i * ngroup

        return ImageLibrary(imageLibrarySharedArray, nimgs = n_imgs, size= self.size, vsize=self.vsize,
                            angles = angles_z, z_shifts=z_shifts, view_group=view_group)

    def afmize(self, amfize_path, pdb, probe_r, probe_a, noise, prefix):

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

        os.system("%s %s.toml" % (amfize_path, prefix))

        return np.loadtxt(prefix + ".tsv").T

    def init_projLibrary_processes(self, imageLibrarySharedArray, zshiftRawArray, n_imgs,
                n_zshifts, zero_cutoff, zshift_range, simulator, pdb,init_zshift, true_zshift):
        global image_library_global
        global z_shifts_global
        global n_zshifts_global
        global zero_cutoff_global
        global zshift_range_global
        global simulator_global
        global pdb_global
        global init_zshift_global
        global true_zshift_global

        image_library_global = frombuffer(imageLibrarySharedArray, dtype=np.float32,
                                   count=len(imageLibrarySharedArray)).reshape(n_imgs,
                                                                                      simulator.size,
                                                                                      simulator.size)
        z_shifts_global = frombuffer(zshiftRawArray, dtype=np.float64,
                                   count=len(zshiftRawArray))
        n_zshifts_global = n_zshifts
        zero_cutoff_global = zero_cutoff
        zshift_range_global = zshift_range
        simulator_global = simulator
        pdb_global = pdb
        init_zshift_global = init_zshift
        true_zshift_global = true_zshift

    def run_projLibrary_process(self, workdata):
        rank =  workdata[0]
        angles = workdata[1]

        rot = pdb_global.copy()
        rot.rotate(angles)

        # set zshift
        if init_zshift_global is None:
            z_shift = -rot.coords[:, 2].min()
        else:
            z_shift = init_zshift_global

        if true_zshift_global:
            for z in range(n_zshifts_global):
                image_library_global[rank*n_zshifts_global + z] = simulator_global.pdb2afm(rot, zshift=z_shift + zshift_range_global[z])
                z_shifts_global[rank * n_zshifts_global + z] = z_shift + zshift_range_global[z]
        else:
            max_zshift = np.max(zshift_range_global)
            img = simulator_global.pdb2afm(rot, zshift=z_shift +max_zshift)

            for z in range(n_zshifts_global):
                img_zshift = img.copy()
                img_zshift[img_zshift >= zero_cutoff_global] += zshift_range_global[z]-max_zshift
                img_zshift[img_zshift < 0.0] = 0.0

                z_shifts_global[rank * n_zshifts_global + z] = z_shift + zshift_range_global[z]

                image_library_global[rank*n_zshifts_global + z] = img_zshift


@njit
def _pdb2afm_grad_njit(coords, modes, psim, pix, size, vsize, beta, r, sigma, zshift):
    dq = np.zeros((modes.shape[0], size,size))
    tmodes = np.ascontiguousarray(modes.transpose(1,0,2))

    n_atoms = coords.shape[0]
    sgmam2 = 1/ (2 * (sigma ** 2))
    bsgm2 = beta/sigma**2
    size2 = size / 2
    for i in range(n_atoms):
        for x, y, a in pix[i]:
            if a:
                mux = (x - size2)*vsize
                muy = (y - size2)*vsize
                expnt = np.exp(
                    -((coords[i,0] - mux)**2 + (coords[i,1] - muy)**2) *sgmam2
                    +(coords[i,2]+r[i] + zshift)/beta
                    - psim[x,y]/beta
                )
                dpsim = np.array([expnt, expnt, expnt])
                dpsim[0] *=  (coords[i,0] - mux) * -(bsgm2)
                dpsim[1] *=  (coords[i,1] - muy) * -(bsgm2)
                dq[:,x,y] +=  np.dot(tmodes[i], dpsim)

    return dq



@njit
def _pdb2afm_njit(coords, size, vsize, pix, beta, r , sigma, zshift):
    n_atoms = coords.shape[0]
    img = np.zeros((size,size))
    sgmam2 = 1/ (2 * (sigma ** 2))
    size2=size / 2
    for i in range(n_atoms):
        for x, y, a in pix[i]:
            if a:
                expnt = np.exp(
                    -((coords[i,0] - (x - size2)*vsize)**2 + (coords[i,1] -  (y - size2)*vsize)**2) *sgmam2
                    +(coords[i,2] + zshift+r[i])/beta
                )
                img[x,y] += expnt
    return beta * np.log(1+ img)

@njit
def get_circle(radius ):
    s = radius
    circle = []
    for x in range(s):
        for y in range(s):
            d = (x-s//2)**2 + (y-s//2)**2
            if d<= (s/2)**2:
                circle.append([x,y])
    return np.array(circle)
@njit
def _select_pixels(coord, size, voxel_size, n_pix_cutoff):
    n_atoms = coord.shape[0]
    threshold = (n_pix_cutoff-1)//2
    circle = get_circle(n_pix_cutoff)
    n_pix = circle.shape[0]
    pix= np.zeros((n_atoms, n_pix, 3), dtype=np.int32)
    warn =False
    for i in range(n_atoms):
        p0 = np.floor(coord[i, :2]/voxel_size -threshold + size/2)
        pix[i,:,:2] =p0+ circle
        for j in range(n_pix):
            if (pix[i,j, 0]< size and pix[i,j, 0]>=0) and (pix[i,j, 1]< size and pix[i,j, 1]>=0):
                pix[i,j,2] = 1
            else:
                warn=True
    if warn :
        pass
        # print("WARNING : Atomic coordinates got outside the box")
    return pix

