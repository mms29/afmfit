import afmfit

import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from os.path import join
from scipy.spatial.transform import Rotation
from scipy.ndimage import map_coordinates
import matplotlib
from sklearn.decomposition import PCA
from umap import UMAP
import os
import tempfile
import shutil
import warnings

def show_stats(fitter):

    mser  = np.sqrt(fitter.flexible_mses[:,0])
    msef  = np.sqrt(np.min(fitter.flexible_mses, axis=1))
    rmsd  = fitter.flexible_rmsds[(np.arange(fitter.nimgs),np.argmin(fitter.flexible_mses, axis=1))]
    def get_pct(vr, vf):
        return -100*vr/vf * (1- (vr/vf))
    print("Best MSE rigid : %.2f (s %.2f)"%(mser.mean(), mser.std()) )
    print("Best MSE flexible : %.2f (s %.2f)"%((msef ).mean(),msef.std()))
    print("MSE decrease : %.2f %%"%(get_pct((mser).mean(),(msef ).mean())))
    print("Best RMSD: %.2f Ang (s %.2f Ang) "%(rmsd.mean(), rmsd.std()))



def align_coords(coords, ref, match=None):
    aligned_coords = np.zeros(coords.shape, dtype=np.float32)
    tmp = ref.copy()
    for i in range(len(coords)):
        tmp.coords = coords[i]
        aligned = tmp.alignMol(ref, idx_matching_atoms=match)
        aligned_coords[i] = aligned.coords
    return aligned_coords

class DimRed:
    def __init__(self, pdb, coords, n_components, method="pca"):
        self.pdb= pdb
        self.method = method
        self.coords = coords
        self.n_components = n_components
        self.n_data = len(coords)

        self.dimred = None
        self.data=None

    def run(self,  **kwargs):

        if self.method == "umap":
            self.dimred = UMAP(n_components=self.n_components, random_state=0,  **kwargs)
        else:
            self.dimred = PCA(n_components=self.n_components,  **kwargs)

        data = []
        for i in range(self.n_data):
            data.append(self.coords[i].flatten())
        self.data = self.dimred.fit_transform(np.array(data))

    @classmethod
    def from_fitter(cls, fitter, n_components, method="pca", **kwargs):
        aligned_coords = align_coords(fitter.flexible_coords, fitter.pdb)
        dimred = cls(fitter.pdb, coords=aligned_coords, n_components=n_components, method=method)
        dimred.run(**kwargs)
        return dimred

    def cluster_linear(self, ax=0, n_points=5, method="max"):
        p = np.zeros((self.n_components, n_points))
        if method == "max":
            p[ax] = np.linspace(self.data[:, ax].min(), self.data[:, ax].max(), n_points)
        elif method == "std":
            p[ax] = np.linspace(self.data[:, ax].mean() - 2 * self.data[:, ax].std(), self.data[:, ax].mean() + 2 * self.data[:, ax].std(),
                                n_points)
        cluster = []
        for i in range(self.n_data):
            cluster.append(np.argmin(np.linalg.norm(p.T - self.data[i], axis=1)) + 1)
        return cluster, p

    def cluster2coords(self, cluster):
        n_points = len(set(cluster))
        outcoords = np.zeros((n_points, self.coords.shape[1], 3))
        for i in range(n_points):
            outcoords[i] = self.coords[np.where((i + 1) == np.array(cluster))[0]].mean(axis=0)
        return outcoords

    def traj2coords(self, traj):
        _, n_points = traj.shape
        outcoords = self.dimred.inverse_transform(traj.T).reshape(n_points, self.pdb.n_atoms, 3)
        return outcoords

    def show_pca_ev(self):
        if self.method == "pca":
            fig, ax = plt.subplots(1, 1, figsize=(5, 3))
            ax.stem(np.arange(1, len(self.dimred.explained_variance_ratio_) + 1), 100 * self.dimred.explained_variance_ratio_)
            ax.set_xlabel("#PC")
            ax.set_ylabel("EV (%) ")
            ax.set_title("Explained variance (%) ")
            fig.show()
        else:
            raise RuntimeError("Not available for UMAP")

    def show(self, cval=None, ax=None, points=None, cname=None, cmap="viridis"):
        if ax is None:
            ax = [0, 1]
        fig, axp = plt.subplots(1, 1, figsize=(5, 3))
        if cval is not None:
            sc = axp.scatter(self.data[:, ax[0]], self.data[:, ax[1]], c=cval, cmap=cmap)
            cbar = fig.colorbar(sc, ax=axp)
            if cname is not None :
                cbar.set_label(cname)
        else:
            axp.scatter(self.data[:, ax[0]], self.data[:, ax[1]])

        if points is not None:
            axp.plot(points[ax[0]], points[ax[1]], "o-", color="red")

        axname = "PC" if self.method=="pca" else "UMAP"
        axp.set_xlabel(axname + str(ax[0]))
        axp.set_ylabel(axname + str(ax[1]))
        fig.show()

    def viewAxisChimera(self, ax=0, avg=True, method="std", n_points=5, align=False, align_ref=None, prefix=None):
        cluster, traj = self.cluster_linear(ax=ax, n_points=n_points, method=method)
        self.show(cval=cluster, points=traj, cname="Clusters")


        if avg:
            coords = self.cluster2coords(cluster)
        else:
            if method == "umap":
                raise RuntimeError("Inverse UMAP not available")
            else:
                coords = self.traj2coords(traj)


        if align:
            if align_ref is None:
                align_ref = self.pdb
            match = self.pdb.matchPDBatoms(align_ref)

        # numpyArr2dcd(coords, tmpdir+"traj.dcd")
        # dimred.pdb.write_pdb(tmpdir+"pdb.dcd")

        with tempfile.TemporaryDirectory() as tmpdir:
            if prefix is not None:
                tmpdir = prefix
            for i in range(n_points):
                tmp = self.pdb.copy()
                tmp.coords = coords[i]
                if align:
                    tmp = tmp.alignMol(align_ref, idx_matching_atoms=match)
                tmp.write_pdb("%straj%i.pdb" % (tmpdir, i + 1))

            with open(tmpdir + "traj.cxc", "w") as f:
                for i in range(n_points):
                    f.write("open %straj%i.pdb \n" % (tmpdir, i + 1))
                f.write("morph #1-%i \n" % (n_points))

            run_chimerax(tmpdir +"traj.cxc")

def afm_reconstruct(stk, angles, shifts, mask=None, order=1, operation=1, vsize=1.0):
    nimg = len(stk)
    size = stk[0].shape[0]
    final = np.ones((size,size, size))
    for i in range(nimg):
        print(i)
        if mask is not None:
            img_in = stk[i]*mask[i]
        else:
            img_in = stk[i]
        vol = img2vol(img_in, size=size, vsize=vsize, shift=shifts[i], mask =mask)
        volrot = rotate_vol(vol,angles[i], order=order)
        if operation == 0:
            final*=volrot
        else:
            final+=volrot
    return final

def rotate_vol(vol, angle, order=2):
    phi = angle[0]
    psi = angle[1]
    the = angle[2]

    # create meshgrid
    dim = vol.shape
    ax = np.arange(dim[0])
    ay = np.arange(dim[1])
    az = np.arange(dim[2])
    coords = np.meshgrid(ax, ay, az)

    # stack the meshgrid to position vectors, center them around 0 by substracting dim/2
    xyz = np.vstack([coords[0].reshape(-1) - float(dim[0]) / 2,  # x coordinate, centered
                     coords[1].reshape(-1) - float(dim[1]) / 2,  # y coordinate, centered
                     coords[2].reshape(-1) - float(dim[2]) / 2])  # z coordinate, centered

    # create transformation matrix
    mat = Rotation.from_euler("ZXZ", np.array(angle), degrees=True).as_matrix()

    # apply transformation
    transformed_xyz = np.dot(mat.T, xyz)

    # extract coordinates
    x = transformed_xyz[0, :] + float(dim[0]) / 2
    y = transformed_xyz[1, :] + float(dim[1]) / 2
    z = transformed_xyz[2, :] + float(dim[2]) / 2

    x = x.reshape((dim[0],dim[1],dim[2]))
    y = y.reshape((dim[0],dim[1],dim[2]))
    z = z.reshape((dim[0],dim[1],dim[2])) # reason for strange ordering: see next line

    # the coordinate system seems to be strange, it has to be ordered like this
    new_xyz = [y,x, z]

    # sample
    volR = map_coordinates(vol, new_xyz, order=order)
    return volR

def img2vol(img, size,vsize, shift, mask=None):
    vol = np.zeros((size,size, size))
    img_center = np.zeros(img.shape)
    if mask is None:
        mask = img != 0.0

    xshift = -int(shift[0]/vsize)
    yshift = -int(shift[1]/vsize)
    zshift = float(shift[2])

    img_center[mask] = img[mask] +  (size/2)*vsize - zshift
    img_center = img_center/vsize

    if xshift>=0: img_center[xshift:] = img_center[:(size-xshift)]
    else: img_center[:(size+xshift)] = img_center[-xshift:]
    if yshift>=0: img_center[:,yshift:] = img_center[:,:(size-yshift)]
    else: img_center[:,:(size+yshift)] = img_center[:,-yshift:]

    xi = np.arange(size)
    _,_, zz = np.meshgrid(xi,xi,xi)
    vol[(zz.transpose(2,1,0) < img_center.T)] = 1.0
    # vol[(zz.transpose(2,1,0) < img_center.T-1)] = 0.0
    vol = vol.transpose(2,1,0)
    return vol
def translate(img, shift):
    dx,dy = shift.astype(int)
    img_out = np.zeros(img.shape)
    size = img.shape[0]
    if dx >=0 and dy >=0:
        img_out[dx:, dy:] = img[:size-dx, :size-dy]
    elif dx <0 and dy <0:
        img_out[:size+dx, :size+dy] = img[-dx:, -dy:]
    elif dx >=0 and dy <0:
        img_out[dx:, :size+dy] = img[:size-dx, -dy:]
    elif dx <0 and  dy >=0:
        img_out[:size+dx, dy:] = img[-dx:, :size-dy]
    return img_out


def get_project_root() -> Path:
    return Path(__file__).parent
def get_tests_root():
    return join(get_project_root(), "../tests")

def get_tests_data():
    return join(get_tests_root(), "tests_data")
def get_tests_tmp():
    return join(get_tests_root(), "tests_tmp")


def get_cc(map1, map2):
    return np.sum(map1 * map2) / np.sqrt(np.sum(np.square(map1)) * np.sum(np.square(map2)))
def get_corr(map1, map2):
    return np.sum((map1-np.mean(map1)) * (map2-np.mean(map2)))

def get_angular_distance_vec(p1, p2):
    return np.rad2deg(np.arccos(np.dot(p1,p2)/(np.linalg.norm(p1)*np.linalg.norm(p2))))


def get_angular_distance(a1, a2):
    R1 = euler2matrix(a1)
    R2 = euler2matrix(a2)
    R = np.dot(R1, R2.T)
    return np.rad2deg(np.arccos((np.trace(R) - 1)/2))


def select_points_in_sphere(points, corr, n_points, threshold, plot =False):

    idx_corr = np.argsort(corr)[::-1]
    n_sel = 1
    candidate_idx = 0
    idx_sel = np.array([candidate_idx])

    while (n_sel < n_points):
        cond = [threshold < angular_distance(xi, points[idx_corr[candidate_idx]]) for xi in points[idx_corr[idx_sel]]]
        if all(cond):
            n_sel += 1
            idx_sel = np.concatenate((idx_sel, np.array([candidate_idx])))
        candidate_idx += 1
        if candidate_idx >= idx_corr.shape[0]:
            print("Warning : number of points too large")
            break

    if plot :
        ax = plt.figure().add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=corr, cmap="jet", vmin=np.min(corr[np.nonzero(corr)]))
        ax.scatter(points[idx_corr[idx_sel], 0], points[idx_corr[idx_sel], 1], points[idx_corr[idx_sel], 2], s=100)
        plt.show()

    return idx_corr[idx_sel]

def get_sphere(angular_dist):
    num_pts = int(np.pi * 10000 * 1 / (angular_dist ** 2))
    angles = np.zeros((num_pts, 3))
    indices = np.arange(0, num_pts, dtype=float) + 0.5
    phi = (np.arccos(1 - 2.0 * indices / num_pts))
    theta = (np.pi * (1 + 5 ** 0.5) * indices)
    for i in range(num_pts):
        R = Rotation.from_euler("yz", np.array([phi[i], theta[i]]), degrees=False).as_matrix()
        angles[i] = matrix2euler(R)
    return angles

def get_sphere_full(angular_dist):
    n_zviews = 360 // angular_dist
    angles = get_sphere(angular_dist)
    num_pts = len(angles)
    new_angles = np.zeros((num_pts * n_zviews, 3))
    for i in range(num_pts):
        for j in range(n_zviews):
            new_angles[i*n_zviews+ j, 0] =j * angular_dist
            new_angles[i*n_zviews+ j, 1] =angles[i,1]
            new_angles[i*n_zviews+ j, 2] =angles[i,2]
    return new_angles


def get_points_from_angles(angles):
    points = np.zeros(angles.shape)
    for i in range(angles.shape[0]):
        R = euler2matrix(angles[i])
        vec1 = np.array([0,0,1])
        points[i] = np.dot(R, vec1)
    return points

def get_sphere_near(angular_dist, near_angle, near_angle_cutoff):
    angles = get_sphere(angular_dist)
    angles_final = [near_angle]
    for i in range(len(angles)):
        if get_angular_distance(near_angle, angles[i]) < near_angle_cutoff:
            angles_final.append(angles[i])
    return np.array(angles_final)

def euler2matrix(angles):
    return Rotation.from_euler("zyz", np.array(angles), degrees=True).as_matrix()

def matrix2euler(A):
    return Rotation.from_matrix(A).as_euler("zyz", degrees=True)

def rotation_matrix_from_vectors(vec2):
    vec1 = np.array([1,0,0])
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def to_mesh(img, vsize, file, upsample=1, truemesh=True):
    import scipy
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=img.max())
    cmap = matplotlib.colormaps.get_cmap('afmhot_r')

    imgup = scipy.ndimage.zoom(img, upsample, order=3)
    vsizeup = vsize/upsample

    size = imgup.shape[0]
    origin= np.array([0.0,0.0,0.0])
    x = np.linspace(-size/2 * vsizeup, size/2 *vsizeup, size)
    y = np.linspace(-size/2 * vsizeup, size/2 *vsizeup, size)

    with open(file , "w") as f:
        if truemesh:
            for i in range(size):
                for j in range(size):
                    color = cmap(1-norm(imgup[i,j]))
                    f.write(".color %.2f %.2f %.2f \n"%(color[0], color[1], color[2]))
                    if j==0:
                        f.write(".m %.2f %.2f %.2f \n" % (x[i], y[j], imgup[i, j]))
                    else:
                        f.write(".d %.2f %.2f %.2f \n" % (x[i], y[j], imgup[i, j]))
            for j in range(size):
                for i in range(size):
                    color = cmap(1 - norm(imgup[i, j]))
                    f.write(".color %.2f %.2f %.2f \n" % (color[0], color[1], color[2]))
                    if i == 0:
                        f.write(".m %.2f %.2f %.2f \n" % (x[i], y[j], imgup[i, j]))
                    else:
                        f.write(".d %.2f %.2f %.2f \n" % (x[i], y[j], imgup[i, j]))
            f.write("\n")

        else:
            for i in range(size):
                for j in range(size):
                    color = cmap(1-norm(imgup[i,j]))
                    f.write(".color %.2f %.2f %.2f \n"%(color[0], color[1], color[2]))

                    f.write(".dot %.2f %.2f %.2f \n"%(x[i], y[j], imgup[i,j]))
                    if j+1<size and i+1< size:
                        f.write(".v %.2f %.2f %.2f %.2f %.2f %.2f \n" % (x[i], y[j], imgup[i, j],x[i], y[j+1], imgup[i, j+1]))
                        f.write(".v %.2f %.2f %.2f %.2f %.2f %.2f \n" % (x[i], y[j], imgup[i, j],x[i+1], y[j], imgup[i+1, j]))
                    f.write("\n")


def check_chimerax():
    return shutil.which("chimerax") is not None


def run_chimerax(args=""):
    if check_chimerax():
        with tempfile.TemporaryDirectory() as tmpdirname:
            script = join(tmpdirname,  "chimerax.sh")
            with open(script, "w") as f:
                f.write("#! ")
                f.write(os.environ.get("SHELL"))
                f.write("\n")
                f.write("chimerax %s"%args)
            os.system("chmod 777 %s && %s"%(script, script))
    else:
        raise RuntimeError("ChimeraX not found")

def get_nolb_path():
    return join(afmfit.__path__[0], join("nolb", "NOLB"))
