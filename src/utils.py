import math
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from scipy.ndimage import binary_erosion, binary_dilation
from pathlib import Path
from os.path import join
from scipy.spatial.transform import Rotation


def get_project_root() -> Path:
    return Path(__file__).parent.parent
def get_tests_root():
    return join(get_project_root(), "tests")

def get_tests_data():
    return join(get_tests_root(), "tests_data")
def get_tests_tmp():
    return join(get_tests_root(), "tests_tmp")

def mask_stk(stk, itere = 1, iterd = 2,threshold = 40.0,sigma = 1.0) :
    mask_stk = np.zeros(stk.shape)
    for i in range(stk.shape[0]):
        g = gaussian(stk[i], sigma=sigma)
        binary_mask = g > threshold
        binary_mask = binary_erosion(binary_mask, iterations=itere)
        binary_mask = binary_dilation(binary_mask, iterations=iterd)
        mask_stk[i] = binary_mask
    return mask_stk

def get_cc(map1, map2):
    return np.sum(map1 * map2) / np.sqrt(np.sum(np.square(map1)) * np.sum(np.square(map2)))
def get_corr(map1, map2):
    return np.sum((map1-np.mean(map1)) * (map2-np.mean(map2)))

def angular_distance(p1, p2):
    return np.rad2deg(np.arccos(np.dot(p1,p2)/(np.linalg.norm(p1)*np.linalg.norm(p2))))

def get_angular_distance(a1, a2):
    R1 = generate_euler_matrix_deg(a1)
    R2 = generate_euler_matrix_deg(a2)
    p = np.ones(3).T
    p1 = np.dot(R1,p)
    p2 = np.dot(R2,p)
    v = np.dot(p1, p2)/(np.linalg.norm(p1)*np.linalg.norm(p2))
    if v >1.0:
        v=1.0
    a = np.arccos(v)
    return np.rad2deg(a)

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
    indices = np.arange(0, num_pts, dtype=float) + 0.5
    phi = (np.arccos(1 - 2 * indices / num_pts))
    theta = (np.pi * (1 + 5 ** 0.5) * indices)
    x = np.array([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]).T
    return x

def get_sphere_near(angular_dist, near_point, near_cutoff):
    points = get_sphere(angular_dist)
    points_final = [near_point]
    for i in range(len(points)):
        if angular_distance(near_point, points[i]) < near_cutoff:
            points_final.append(points[i])
    return np.array(points_final)


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
    a,b,c = np.deg2rad(angles)
    cos = np.cos
    sin = np.sin
    R = np.array([[ cos(c) *  cos(b) * cos(a) -  sin(c) * sin(a), cos(c) * cos(b) * sin(a) +  sin(c) * cos(a), -cos(c) * sin(b)],
                  [- sin(c) *  cos(b) * cos(a) - cos(c) * sin(a), - sin(c) * cos(b) * sin(a) + cos(c) * cos(a), sin(c) * sin(b)],
                  [sin(b) * cos(a), sin(b) * sin(a), cos(b)]])
    return R

def matrix2eulerAngles(A):
    abs_sb = np.sqrt(A[0, 2] * A[0, 2] + A[1, 2] * A[1, 2])
    if (abs_sb > 16*np.exp(-5)):
        gamma = math.atan2(A[1, 2], -A[0, 2])
        alpha = math.atan2(A[2, 1], A[2, 0])
        if (abs(np.sin(gamma)) < np.exp(-5)):
            sign_sb = np.sign(-A[0, 2] / np.cos(gamma))
        else:
            if np.sin(gamma) > 0:
                sign_sb = np.sign(A[1, 2])
            else:
                sign_sb = -np.sign(A[1, 2])
        beta = math.atan2(sign_sb * abs_sb, A[2, 2])
    else:
        if (np.sign(A[2, 2]) > 0):
            alpha = 0
            beta  = 0
            gamma = math.atan2(-A[1, 0], A[0, 0])
        else:
            alpha = 0
            beta  = np.pi
            gamma = math.atan2(A[1, 0], -A[0, 0])
    gamma = np.rad2deg(gamma)
    beta  = np.rad2deg(beta)
    alpha = np.rad2deg(alpha)
    return alpha, beta, gamma

def rotation_matrix_from_vectors(vec2):
    vec1 = np.array([1,0,0])
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

