import matplotlib.pyplot as plt
import numpy as np
from numba import njit
import time
from src.utils import get_sphere_near, get_sphere, generate_euler_matrix_deg, get_points_from_angles, get_sphere_full
from . import AFMIZE_PATH
import os
from src.viewer import viewAFM
import tqdm
from multiprocessing.sharedctypes import RawArray
from numpy import frombuffer
from multiprocessing import Pool

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
            zshift = -pdb.coords[:, 2].min()
        afm = _pdb2afm_njit(coords=pdb.coords, size=self.size, vsize=self.vsize, pix=pix, beta=self.beta, r=r ,
                 sigma=self.sigma, zshift=zshift)
        return afm

    def pdb2afm_grad(self, pdb,nma, psim, zshift):
        r = pdb.get_radius()
        pix = _select_pixels(pdb.coords, size=self.size, voxel_size=self.vsize, n_pix_cutoff=self.n_pix_cutoff)

        grad = _pdb2afm_grad_njit(coords=pdb.coords, modes=nma.linear_modes, psim=psim, pix=pix, size=self.size,
            zshift=zshift, vsize=self.vsize, beta=self.beta, r=r, sigma=self.sigma)
        return grad

    def get_projection_library(self, pdb, init_zshift = None,  zshift_range=None, angular_dist=10,
                           verbose=True, near_angle=None, near_angle_cutoff=30):
        ZERO_CUTOFF =5.0 # Angstrom

        # Compute directions views
        if near_angle is not None:  angles = get_sphere_near(angular_dist, near_angle, near_angle_cutoff)
        else:                       angles = get_sphere(angular_dist)

        # Initiate arrays
        n_angles = len(angles)
        if zshift_range is None:
            n_zshift = 1
            zshift_range = [0.0]
        else:
            n_zshift = len(zshift_range)
        n_img = n_angles * n_zshift
        image_library = np.zeros((n_img,  self.size, self.size))

        z_shifts = np.zeros(n_img)
        all_angles = np.zeros((n_img, 3))

        time_rot = 0.0
        time_proj = 0.0
        time_zsh = 0.0

        # Loop over all viewing directions
        dtime = time.time()
        for i in range(n_angles):
            if verbose:
                if int((i + 1) % (n_angles / 10)) == 0:
                    print("%i %%" % int(100 * (i + 1) / n_angles))

            # Rotate coordinates
            dt = time.time()
            rot_pdb = pdb.copy()
            rot_pdb.rotate(angles[i])
            if init_zshift is None:
                z_shift = -rot_pdb.coords[:, 2].min()
            else:
                z_shift = init_zshift
            time_rot += time.time() - dt

            # Simulate images
            dt = time.time()
            img = self.pdb2afm(rot_pdb, zshift=z_shift)
            time_proj += time.time() - dt

            # Simulate images in Z direction
            dt = time.time()
            for z in range(n_zshift):
                img_zshift = img.copy()
                img_zshift[img_zshift >= ZERO_CUTOFF] += zshift_range[z]
                if zshift_range[z] < 0:
                    img_zshift[img_zshift < 0.0] = 0.0
                image_library[i * n_zshift + z] = img_zshift
                z_shifts[i * n_zshift + z] = zshift_range[z] + z_shift
                all_angles[i * n_zshift + z] = angles[i]
            time_zsh += time.time() - dt

        shifts = np.zeros((n_img, 3))
        shifts[:, 2] = z_shifts

        if verbose:
            print("\t TIME_TOTAL => 100 %% ;  %.4f s" % (time.time() - dtime))
            print("\t  rot ;  %.4f s" % (time_rot))
            print("\t  proj ;  %.4f s" % (time_proj))
            print("\t  zsh ;  %.4f s" % (time_zsh))

        return ImageSet(image_library, angles = all_angles, shifts=shifts)

    def get_projection_library_full(self, pdb, angular_dist, init_zshift = None, verbose=True, zshift_range=None):
        ZERO_CUTOFF = 5.0  # Angstrom

        # Compute directions views
        angles = get_sphere_full(angular_dist)

        # Initiate arrays
        n_angles, _ = angles.shape
        n_zshifts = len(zshift_range)
        n_imgs = n_angles*n_zshifts
        image_library = np.zeros((n_imgs,  self.size, self.size), dtype=np.float32)
        z_shifts = np.zeros(n_imgs)
        angles_z = np.zeros((n_imgs,3))

        # Loop over all viewing directions
        dtime = time.time()
        for i in tqdm.tqdm(range(n_angles)):
            # Rotate coordinates
            rot_pdb = pdb.copy()
            rot_pdb.rotate(angles[i])

            # set zshift
            if init_zshift is None: z_shift = -rot_pdb.coords[:, 2].min()
            else: z_shift = init_zshift

            # Simulate images
            img = self.pdb2afm(rot_pdb, zshift = z_shift)

            for z in range(n_zshifts):
                img_zshift = img.copy()
                img_zshift[img_zshift >= ZERO_CUTOFF] += zshift_range[z]
                if zshift_range[z] < 0:
                    img_zshift[img_zshift < 0.0] = 0.0

                image_library[i*n_zshifts + z] = img_zshift
                z_shifts[i*n_zshifts + z] = z_shift + zshift_range[z]
                angles_z[i*n_zshifts + z] = angles[i]

        if verbose:
            print("\t TIME_TOTAL => 100 %% ;  %.4f s" % (time.time() - dtime))

        return ImageLibrary(image_library, angles = angles_z, z_shifts=z_shifts)

    def get_projection_library_full_pool(self, n_cpu, pdb, angular_dist, init_zshift = None, verbose=True, zshift_range=None):
        ZERO_CUTOFF = 5.0  # Angstrom

        # Compute directions views
        angles = get_sphere_full(angular_dist)

        # Initiate arrays
        n_angles, _ = angles.shape
        n_zshifts = len(zshift_range)
        n_imgs = n_angles*n_zshifts
        z_shifts = np.zeros(n_imgs)
        angles_z = np.zeros((n_imgs,3))


        workdata=[]

        # Loop over all viewing directions
        dtime = time.time()
        for i in tqdm.tqdm(range(n_angles), desc="Rotate Model"):
            # Rotate coordinates
            rot_pdb = pdb.copy()
            rot_pdb.rotate(angles[i])

            # set zshift
            if init_zshift is None: z_shift = -rot_pdb.coords[:, 2].min()
            else: z_shift = init_zshift
            for z in range(n_zshifts):
                z_shifts[i*n_zshifts + z] = z_shift + zshift_range[z]
                angles_z[i*n_zshifts + z] = angles[i]

            workdata.append([
            i ,
            rot_pdb.coords,
            z_shift ,
            ])

        imageLibrarySharedArray = RawArray("f", n_imgs*self.size*self.size)
        image_library = frombuffer(imageLibrarySharedArray, dtype=np.float32,
                                   count=len(imageLibrarySharedArray)).reshape(n_imgs, self.size, self.size)


        p = Pool(n_cpu, initializer=init_pool_processes, initargs=(imageLibrarySharedArray,n_imgs,
                n_zshifts, ZERO_CUTOFF, zshift_range, self, pdb))
        for _ in tqdm.tqdm(p.imap_unordered(run_pool_process, workdata), total=len(workdata), desc="Project Library"):
            pass

        del workdata

        if verbose:
            print("\t TIME_TOTAL => 100 %% ;  %.4f s" % (time.time() - dtime))

        return ImageLibrary(image_library, angles = angles_z, z_shifts=z_shifts)



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

def run_pool_process(workdata):
    rank =  workdata[0]
    coords = workdata[1]
    zshift = workdata[2]

    image_library = frombuffer(imageLibrarySharedArrayProcess, dtype=np.float32,
                               count=len(imageLibrarySharedArrayProcess)).reshape(n_imgs_global, simulator_global.size,
                                                                                  simulator_global.size)
    # Simulate images
    rot = pdb_global.copy()
    rot.coords = coords

    max_zshift = np.max(zshift_range_global)
    img = simulator_global.pdb2afm(rot, zshift=zshift +max_zshift)

    for z in range(n_zshifts_global):
        img_zshift = img.copy()
        img_zshift[img_zshift >= zero_cutoff_global] += zshift_range_global[z]-max_zshift
        img_zshift[img_zshift < 0.0] = 0.0

        image_library[rank*n_zshifts_global + z] = img_zshift
def init_pool_processes(imageLibrarySharedArray, n_imgs,
            n_zshifts, zero_cutoff, zshift_range, simulator, pdb):
    global imageLibrarySharedArrayProcess
    global n_imgs_global
    global n_zshifts_global
    global zero_cutoff_global
    global zshift_range_global
    global simulator_global
    global pdb_global
    imageLibrarySharedArrayProcess = imageLibrarySharedArray
    n_imgs_global = n_imgs
    n_zshifts_global = n_zshifts
    zero_cutoff_global = zero_cutoff
    zshift_range_global = zshift_range
    simulator_global = simulator
    pdb_global = pdb

@njit
def calculate_norm_njit(imgs):
    nimg = imgs.shape[0]
    norm2 = np.zeros(nimg)
    for i in range(nimg):
        norm2[i] = np.sum(np.square(imgs[i]))
    return norm2

class ImageLibrary:

    def __init__(self, imgs, angles=None, z_shifts=None):
        self.imgs = imgs
        self.nimgs = imgs.shape[0]
        self.angles = angles
        self.z_shifts = z_shifts
        self.norm2 = self.calculate_norm(imgs)

    def calculate_norm(self, imgs):
        return calculate_norm_njit(imgs)
    def show(self):
        viewAFM(self.imgs, interactive=True)

    def show_sphere(self):
        points = get_points_from_angles(self.angles)
        points_all = get_points_from_angles(get_sphere(5))
        ax = plt.figure().add_subplot(111, projection='3d')
        ax.scatter(points_all[:, 0], points_all[:, 1], points_all[:, 2], c="grey", s=1)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])
        plt.show()

class ImageSet:

    def __init__(self, imgs, angles=None, shifts=None):
        self.imgs = imgs
        self.nimg = len(imgs)
        self.angles = angles
        self.shifts = shifts

    def show(self):
        viewAFM(self.imgs, interactive=True)

    def show_sphere(self):
        points = get_points_from_angles(self.angles)
        points_all = get_points_from_angles(get_sphere(5))
        ax = plt.figure().add_subplot(111, projection='3d')
        ax.scatter(points_all[:, 0], points_all[:, 1], points_all[:, 2], c="grey", s=1)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])
        plt.show()


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
        print("WARNING : Atomic coordinates got outside the box")
    return pix

