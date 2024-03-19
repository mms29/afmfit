import math
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from os.path import join
from scipy.spatial.transform import Rotation
from scipy.ndimage import map_coordinates
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
    return Path(__file__).parent.parent
def get_tests_root():
    return join(get_project_root(), "tests")

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
    R1 = generate_euler_matrix_deg(a1)
    R2 = generate_euler_matrix_deg(a2)
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
        angles[i] = matrix2eulerAngles(R)
    return angles

# def get_sphere_full(angular_dist):
#     n_zviews = 360 // angular_dist
#     angles = get_sphere(angular_dist)
#     num_pts = len(angles)
#     new_angles = np.zeros((num_pts, n_zviews, 3))
#     for i in range(num_pts):
#         R1 = generate_euler_matrix_deg(angles[i])
#         for j in range(n_zviews):
#             R2 = generate_euler_matrix_deg([j * angular_dist, 0.0,0.0])
#             R3 = np.dot(R2, R1)
#             new_angles[i, j, :] =matrix2eulerAngles(R3)
#     return new_angles

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
        R = generate_euler_matrix_deg(angles[i])
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


# def generate_euler_matrix(angles):
#     a, b, c = angles
#     cos = np.cos
#     sin = np.sin
#     R = np.array([[ cos(c) *  cos(b) * cos(a) -  sin(c) * sin(a), cos(c) * cos(b) * sin(a) +  sin(c) * cos(a), -cos(c) * sin(b)],
#                   [- sin(c) *  cos(b) * cos(a) - cos(c) * sin(a), - sin(c) * cos(b) * sin(a) + cos(c) * cos(a), sin(c) * sin(b)],
#                   [sin(b) * cos(a), sin(b) * sin(a), cos(b)]])
#     return R
# def generate_euler_matrix_deg(angles):
# # def matrix_from_euler(angles):
#     return Rotation.from_euler("zyz", -np.array(angles), degrees=True).as_matrix()
#
#
# def matrix2eulerAngles(A):
# # def euler_from_matrix(A):
#     return Rotation.from_matrix(A.T).as_euler("zyz", degrees=True)[::-1]

def generate_euler_matrix_deg(angles):
# def matrix_from_euler(angles):
    return Rotation.from_euler("zyz", np.array(angles), degrees=True).as_matrix()


def matrix2eulerAngles(A):
# def euler_from_matrix(A):
    return Rotation.from_matrix(A).as_euler("zyz", degrees=True)
#
# def generate_euler_matrix_deg(angles):
#     a,b,c = np.deg2rad(angles)
#     cos = np.cos
#     sin = np.sin
#     R = np.array([[ cos(c) *  cos(b) * cos(a) -  sin(c) * sin(a), cos(c) * cos(b) * sin(a) +  sin(c) * cos(a), -cos(c) * sin(b)],
#                   [- sin(c) *  cos(b) * cos(a) - cos(c) * sin(a), - sin(c) * cos(b) * sin(a) + cos(c) * cos(a), sin(c) * sin(b)],
#                   [sin(b) * cos(a), sin(b) * sin(a), cos(b)]])
#     return R
#
# def matrix2eulerAngles(A):
#     abs_sb = np.sqrt(A[0, 2] * A[0, 2] + A[1, 2] * A[1, 2])
#     if (abs_sb > 16*np.exp(-5)):
#         gamma = math.atan2(A[1, 2], -A[0, 2])
#         alpha = math.atan2(A[2, 1], A[2, 0])
#         if (abs(np.sin(gamma)) < np.exp(-5)):
#             sign_sb = np.sign(-A[0, 2] / np.cos(gamma))
#         else:
#             if np.sin(gamma) > 0:
#                 sign_sb = np.sign(A[1, 2])
#             else:
#                 sign_sb = -np.sign(A[1, 2])
#         beta = math.atan2(sign_sb * abs_sb, A[2, 2])
#     else:
#         if (np.sign(A[2, 2]) > 0):
#             alpha = 0
#             beta  = 0
#             gamma = math.atan2(-A[1, 0], A[0, 0])
#         else:
#             alpha = 0
#             beta  = np.pi
#             gamma = math.atan2(A[1, 0], -A[0, 0])
#     gamma = np.rad2deg(gamma)
#     beta  = np.rad2deg(beta)
#     alpha = np.rad2deg(alpha)
#     return alpha, beta, gamma

def rotation_matrix_from_vectors(vec2):
    vec1 = np.array([1,0,0])
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

