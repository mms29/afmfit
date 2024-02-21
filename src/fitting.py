import numpy as np
import time
import polarTransform
from numba import njit
from src.utils import generate_euler_matrix_deg, matrix2eulerAngles
import matplotlib.pyplot as plt

class AFMFitting:
    def __init__(self, pdb, img, simulator, nma=None, target_pdb=None):
        self.pdb = pdb
        self.img = img
        self.simulator = simulator
        self.nma = nma
        if target_pdb is None:
            self.target_pdb = pdb
        else:
            self.target_pdb= target_pdb

    def projection_matching(self, image_library, angular_dist=10, max_shift_search=5, verbose=True, plot = False):
        dtime = time.time()

        # Get angle size for rotational matching
        if angular_dist >= 20:        angleSize = 2 ** 4
        elif 20 > angular_dist >= 10: angleSize = 2 ** 5
        elif 10 > angular_dist >= 5:  angleSize = 2 ** 6
        elif 5 > angular_dist >= 2:   angleSize = 2 ** 7
        else:                         angleSize = 2 ** 8

        imageSize = self.img.shape[0]
        radiusSize = imageSize // 2
        num_pts = image_library.nimg

        # convert image to fit into polar coordinates and calculate the center of mass
        g_img = self.img
        g_com = get_com(g_img)
        g_pcs, _ = polarTransform.convertToPolarImage(g_img.T, radiusSize=radiusSize, angleSize=angleSize, center=g_com)

        # outputs
        angles = np.zeros((num_pts, 3))
        shifts = np.zeros((num_pts, 3))
        R = np.zeros((num_pts, 3, 3))
        mse = np.zeros(num_pts)

        for i in range(num_pts):
            if verbose:
                if int((i + 1) % (num_pts / 10)) == 0:
                    print("%i %%" % int(100 * (i + 1) / num_pts))

            # rotational matching of each image
            angle_rm, shift_rm, min_mse = _rotational_matching_mse(g_pcs=g_pcs, f_imgs=image_library.imgs[i], g_com=g_com,
                                                                  imageSize=imageSize, angleSize=angleSize,
                                                                  radiusSize=radiusSize,
                                                                  MAX_SHIFT_SEARCH=max_shift_search,
                                                                  zshift_list=image_library.zshift_range)

            # get the min value of MSE
            mse[i] = min_mse

            # Find the correct rotation and translation corresponding to the rotational matching
            R2 = generate_euler_matrix_deg(np.array([-angle_rm, 0.0, 0.0]))
            Rf = np.dot(R2, image_library.rotations[i])
            shift = np.dot(R2, -(np.array([shift_rm[0] * self.simulator.vsize, shift_rm[1] * self.simulator.vsize, -shift_rm[2]])))
            rot, tilt, psi = matrix2eulerAngles(Rf)
            angles[i] = np.array([rot, tilt, psi])
            shifts[i] = shift
            R[i] = Rf

        if plot:
            fig, ax = plt.subplots(3, 5, figsize=(20, 15))
            for i in range(3):
                min_mse = np.argsort(mse)[i]
                out_coords = np.dot(R[min_mse], self.pdb.coords.T).T
                out_coords += shifts[min_mse]
                pdbt = self.pdb.copy()
                pest = self.simulator.pdb2afm(pdbt, zshift = image_library.zshifts[min_mse])
                ax[i, 0].imshow(pest.T, cmap="afmhot", origin="lower")
                ax[i, 1].imshow(self.img.T, cmap="afmhot", origin="lower")
                ax[i, 2].imshow(np.square(self.img.T - pest.T), cmap="jet", origin="lower")
                ax[i, 3].plot(pest[imageSize // 2], color="tab:red")
                ax[i, 3].plot(self.img[imageSize // 2], color="tab:blue")
                ax[i, 4].plot(pest[:, imageSize // 2], color="tab:red")
                ax[i, 4].plot(self.img[:, imageSize // 2], color="tab:blue")
                ax[i, 0].set_title('Fitted')
                ax[i, 1].set_title('Input')
                ax[i, 2].set_title('Diff')

            ax = plt.figure().add_subplot(111, projection='3d')
            ax.scatter(image_library.directions[:, 0],
                       image_library.directions[:, 1],
                       image_library.directions[:, 2], c=mse, cmap="jet_r", vmin=mse[np.nonzero(mse)].min())
            plt.show()

        if verbose:
            print("\t TIME_TOTAL => %.4f s" % (time.time()-dtime))

        return angles, shifts, mse

    def fit_nma(self, n_iter, gamma=50, gamma_rigid=5, fig=None, mask=None, verbose=True, plot=False, q_init=None,
                zshift = None, pdb=None, nma=None):
        dtimetot = time.time()

        # Inputs
        if zshift is None:
            zshift=0.0
        if nma is None:
            nma=self.nma
        if pdb is None:
            pdb=self.pdb
        psim = self.simulator.pdb2afm(pdb, zshift = zshift)
        coords = pdb.coords
        size = self.simulator.size
        pexp = self.img

        # define outputs
        dcd = np.array([coords])
        mse = np.array([np.linalg.norm(psim - pexp)])
        rmsd = np.array([pdb.getRMSD(self.target_pdb, align=True)])

        # Set normal mode amplitudes
        qt = np.zeros((n_iter + 1, nma.nmodes_total))
        if q_init is not None:
            qt[0, :] = q_init.copy()[:]

        # Define regularization
        regul = np.zeros( nma.nmodes_total)
        regul[:6] = 1/(gamma_rigid**2)
        regul[6:] = 1/(gamma**2)
        linear_modes_line =  nma.linear_modes.reshape(nma.nmodes_total, nma.natoms * 3)
        r_matrix = np.dot(linear_modes_line, linear_modes_line.T) * regul

        if plot:
            plt.ion()
            if fig is not None:
                figi = fig
                ax = fig.get_axes()
                ax[0].clear()
                ax[2].clear()
                im1 = ax[0].imshow(psim.T, cmap="afmhot", origin="lower")
                im2 = ax[2].imshow(np.square(psim.T - self.img.T), cmap="afmhot", origin="lower")
                ax[0].set_title("Fitted")
                ax[2].set_title("Diff")
                ax[5].clear()
                ax[6].clear()
                ax[5].plot(psim.T[size // 2])
                ax[5].plot(pexp.T[size // 2])
                ax[6].plot(psim.T[:, size // 2])
                ax[6].plot(pexp.T[:, size // 2])
                figi.canvas.draw()
                figi.canvas.flush_events()

            else:
                figi, ax = plt.subplots(1, 7, figsize=(20, 4))
                im1 = ax[0].imshow(psim.T, cmap="afmhot", origin="lower")
                ax[1].imshow(pexp.T, cmap="afmhot", origin="lower")
                im2 = ax[2].imshow(np.square(psim.T - pexp.T), cmap="afmhot", origin="lower")
                ax[3].plot(mse)
                ax[4].plot(rmsd)
                ax[5].plot(psim.T[size // 2])
                ax[5].plot(pexp.T[size // 2])
                ax[6].plot(psim.T[:, size // 2])
                ax[6].plot(pexp.T[:, size // 2])
                ax[0].set_title("Fitted")
                ax[1].set_title("Target")
                ax[2].set_title("Diff")
                ax[3].set_title("MSE")
                ax[4].set_title("RMSD")

        for i in range(n_iter):

            # Apply NMA amplitudes to the PDB
            fitted = nma.applySpiralTransformation(eta=qt[i], pdb=pdb)

            # Simulate the new image
            try:
                psim = self.simulator.pdb2afm(fitted, zshift=zshift)
            except RuntimeError:
                return dcd, np.ones(n_iter + 1) * mse[0], np.ones(n_iter + 1) * rmsd[0], np.zeros(nma.nmodes_total)

            # Compute the gradient
            dq = self.simulator.pdb2afm_grad(pdb=fitted,nma=nma, psim=psim, zshift=zshift)

            # Compute the new NMA amplitudes
            q_est = q_estimate(psim, pexp, dq,r_matrix=r_matrix, q0=qt[i], mask=mask)
            qt[i + 1] = q_est + qt[i]

            # Update outputs
            dcd = np.concatenate((dcd, np.array([coords])))
            mse = np.concatenate((mse, np.array([np.linalg.norm(psim - pexp)])))
            rmsd = np.concatenate((rmsd, np.array([fitted.getRMSD(self.target_pdb, align=True)])))

            if verbose:
                print("Iter %i = %.2f ; %.2f" % (i, mse[-1], rmsd[-1]))
                print(qt[i + 1])

            if plot:
                if fig is not None:
                    im1.set_data(psim.T)
                    im2.set_data(np.square(psim.T - pexp.T))
                    ax[5].clear()
                    ax[6].clear()
                    ax[5].plot(psim.T[size // 2])
                    ax[5].plot(pexp.T[size // 2])
                    ax[6].plot(psim.T[:, size // 2])
                    ax[6].plot(pexp.T[:, size // 2])
                    figi.canvas.draw()
                    figi.canvas.flush_events()
                else:
                    im1.set_data(psim.T)
                    im2.set_data(np.square(psim.T - pexp.T))
                    ax[3].clear()
                    ax[4].clear()
                    ax[5].clear()
                    ax[6].clear()
                    ax[3].plot(mse)
                    ax[4].plot(rmsd)
                    ax[5].plot(psim.T[size // 2])
                    ax[5].plot(pexp.T[size // 2])
                    ax[6].plot(psim.T[:, size // 2])
                    ax[6].plot(pexp.T[:, size // 2])
                    ax[3].set_title("MSE")
                    ax[4].set_title("RMSD")
                    figi.canvas.draw()
                    figi.canvas.flush_events()

        if verbose:
            print("\t TIME_TOTAL   => %.4f s" % (time.time() - dtimetot))

        return dcd, mse, rmsd, qt


    def fit_nma_rotations(self, solver_iter, gamma, gamma_rigid,
                max_shift_search, angular_dist, near_cutoff, n_points, mask =None,
                plot = True, verbose = False, zshift_range=0.0, zshift_points=1):

        # Inputs
        maxiter = len(angular_dist)
        psim = self.simulator.pdb2afm(self.pdb)
        pexp = self.img
        pdb = self.pdb
        fitted = pdb.copy()
        size = self.simulator.size

        # Output arrays
        mses = np.array([np.linalg.norm(psim- pexp)])
        rmsds = np.array([pdb.getRMSD(self.target_pdb, align=True)])
        dcds = np.array([pdb.coords])
        q_est = np.zeros(self.nma.nmodes_total)
        zshift_est=0.0

        if plot:
            plt.ion()
            fig, ax = plt.subplots(1, 7, figsize=(20, 4))
            ax[0].imshow(psim.T, origin="lower", cmap="afmhot")
            ax[1].imshow(pexp.T, origin="lower", cmap="afmhot")
            ax[2].imshow(np.square(pexp.T - psim.T), origin="lower", cmap="afmhot")
            ax[3].plot(mses)
            ax[4].plot(rmsds)
            ax[5].plot(psim.T[size//2])
            ax[5].plot(pexp.T[size//2])
            ax[6].plot(psim.T[:, size//2])
            ax[6].plot(pexp.T[:, size//2])
            ax[0].set_title("Fitted")
            ax[1].set_title("Target")
            ax[2].set_title("Diff")
            ax[3].set_title("MSE")
            ax[4].set_title("RMSD")
            fig.canvas.draw()
            fig.canvas.flush_events()
        else:
            fig = None

        for ii in range(maxiter):
            print("Iteration : %i"%ii)
            zshift_list = np.linspace(zshift_est - zshift_range[ii], zshift_est + zshift_range[ii], zshift_points[ii])

            if ii == 0:
                image_library= self.simulator.get_projection_library(fitted,
                                        zshift_range=zshift_list, angular_dist=angular_dist[ii], verbose=verbose)
            else:
                image_library= self.simulator.get_projection_library(fitted, near_point=near_point,
                    near_cutoff=near_cutoff[ii], zshift_range=zshift_list, angular_dist=angular_dist[ii], verbose=verbose)

            angles, shifts, mse = self.projection_matching(image_library=image_library, angular_dist=angular_dist[ii],
                    max_shift_search=max_shift_search, verbose=verbose)

            if n_points[ii] <= len(mse):
                min_mses = np.argsort(mse)[:n_points[ii]]
            else:
                min_mses = np.argsort(mse)
            mse_sel = []
            rmsd_sel = []
            q_sel = []
            dcd_sel=[]
            for sel in range(len(min_mses)):
                # Set the shifts
                shifti = shifts[min_mses[sel]]  + np.array([0,0,image_library.zshifts[min_mses[sel]]])

                # rotate NMA
                rot_nma = self.nma.transform(angle=angles[min_mses[sel]], shift=shifti)
                candidate = rot_nma.pdb

                # Fit NMA
                dcd, mse, rmsd, q = self.fit_nma(n_iter=solver_iter, gamma=gamma, gamma_rigid=gamma_rigid,
                                                 fig=fig, mask=mask, verbose=verbose, plot=plot,
                                                 q_init=q_est, zshift = 0.0, pdb=candidate, nma=rot_nma)
                mse_sel.append(mse)
                rmsd_sel.append(rmsd)
                q_sel.append(q)
                dcd_sel.append(dcd)
            mse_sel = np.array(mse_sel)
            q_sel = np.array(q_sel) #[sel, solver iter, nmodes]
            rmsd_sel = np.array(rmsd_sel)

            min_mse = np.unravel_index(np.argmin(mse_sel), mse_sel.shape) #[n_sel, solver_iter]
            q_est = q_sel[min_mse]
            near_point = image_library.directions[min_mses[min_mse[0]]]
            shift_est = shifts[min_mses[min_mse[0]]]
            zshift_est = shift_est[2]
            angle_est = angles[min_mses[min_mse[0]]]

            mses = np.concatenate((mses, mse_sel[min_mse[0]]))
            rmsds = np.concatenate((rmsds, rmsd_sel[min_mse[0]]))
            dcds = np.concatenate((dcds, dcd_sel[min_mse[0]]))

            fitted = self.nma.applySpiralTransformation(eta=q_est, pdb=pdb)
            fitted_rot = fitted.copy()
            fitted_rot.rotate(angle_est)
            fitted_rot.translate(shift_est)
            psim = self.simulator.pdb2afm(fitted_rot, zshift=image_library.zshifts[min_mses[min_mse[0]]])

            if plot:
                ax[0].clear()
                ax[0].imshow(psim.T, origin="lower", cmap="afmhot")
                ax[2].clear()
                ax[2].imshow(np.square(psim.T - pexp.T), origin="lower", cmap="afmhot")
                ax[3].clear()
                ax[4].clear()
                for sel in range(len(mse_sel)):
                    ax[3].plot(np.arange(len(mses) -( solver_iter+1), len(mses)), mse_sel[sel],"o-", color="grey")
                    ax[4].plot(np.arange(len(mses) -( solver_iter+1), len(mses)), rmsd_sel[sel],"o-", color="grey")
                ax[3].plot(mses)
                ax[4].plot(rmsds)
                ax[3].plot(len(mses) - (solver_iter + 1) + min_mse[1], mse_sel[min_mse], "o", color="red")
                ax[4].plot(len(mses) - (solver_iter + 1) + min_mse[1], rmsd_sel[min_mse], "o", color="red")
                ax[5].clear()
                ax[6].clear()
                ax[5].plot(psim.T[size // 2])
                ax[5].plot(pexp.T[size // 2])
                ax[6].plot(psim.T[:, size // 2])
                ax[6].plot(pexp.T[:, size // 2])
                ax[0].set_title("Fitted")
                ax[1].set_title("Target")
                ax[2].set_title("Diff")
                ax[3].set_title("MSE")
                ax[4].set_title("RMSD")
                fig.canvas.draw()
                fig.canvas.flush_events()

        fitted.rotate(angle_est)
        fitted.translate(shift_est)
        fitted.translate(np.array([0,0,image_library.zshifts[min_mses[min_mse[0]]]]))

        dcds = np.concatenate((dcds, [fitted.coords]))

        return dcds, mses, rmsds, q_est, psim


def get_com(img):
    N = img.shape[0]
    com_x = np.sum(np.arange(N) *np.sum(img, axis=1)) / np.sum(img)
    com_y = np.sum(np.arange(N) *np.sum(img, axis=0)) / np.sum(img)
    return np.array([com_x, com_y])

@njit
def corr_sum_njit(f_pcs_fft, g_pcs_fft, angleSize, radiusSize):
    corr_fourier = np.zeros((angleSize), dtype=np.complex128)
    for r in range(radiusSize):
        corr_fourier += r * g_pcs_fft[:, r] * np.conjugate(f_pcs_fft[:, r])
    return corr_fourier


def get_corr_fourier(f_pcs_fft, g_pcs_fft, angleSize, radiusSize):
    corr_fourier = corr_sum_njit(f_pcs_fft, g_pcs_fft, angleSize, radiusSize)
    corr = np.fft.ifftn(corr_fourier).real

    argmax_corr = np.argmax(corr)
    max_corr = corr[argmax_corr]
    return  (argmax_corr*np.pi*2)/angleSize, max_corr
def get_mse_img(g_pcs, f_pcs,angleSize):
    mse = np.zeros(angleSize)
    for i in range(angleSize):
        mse[i] = np.linalg.norm(g_pcs - np.roll(f_pcs, i, axis=0) )
    min_mse = np.argmin(mse)
    return (min_mse*np.pi*2)/angleSize, mse[min_mse]

def rotational_matching(g_pcs_fft, f_img, g_com, f_com, N, angleSize, radiusSize, MAX_SHIFT_SEARCH = 10):
    N_SHIFT_SEARCH = MAX_SHIFT_SEARCH*2 - 1
    curr_corr = 0.0
    curr_angle = 0.0
    curr_com = f_com
    max_corrs = np.zeros((N_SHIFT_SEARCH, N_SHIFT_SEARCH))
    for x in range(N_SHIFT_SEARCH):
        xi = x - MAX_SHIFT_SEARCH + 1
        for y in range(N_SHIFT_SEARCH):
            yi = y - MAX_SHIFT_SEARCH + 1
            tmp_com = f_com + np.array([xi,yi])
            f_pcs, _ = polarTransform.convertToPolarImage(f_img.T, radiusSize=radiusSize, angleSize=angleSize, center=tmp_com)
            f_pcs_fft = np.fft.fft(f_pcs, axis=0)

            angle, max_corr = get_corr_fourier(f_pcs_fft, g_pcs_fft, angleSize, radiusSize)
            max_corrs[x,y] = max_corr
            if max_corr> curr_corr:
                curr_corr=max_corr
                curr_angle = angle
                curr_com = tmp_com
    theta = curr_angle
    mat = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    g_com_rot = np.dot(mat.T, g_com - N//2) + N//2
    shift_est = curr_com-g_com_rot

    return np.rad2deg(theta), shift_est, curr_corr, #g_com_rot, g_com, curr_com, max_corrs


def _rotational_matching_mse(g_pcs, f_imgs, g_com, imageSize, angleSize, radiusSize, zshift_list=None, MAX_SHIFT_SEARCH = 5):
    N_SHIFT_SEARCH = MAX_SHIFT_SEARCH*2 - 1
    if zshift_list is None:
        Z_SHIFT_SEARCH = 1
    else:
        Z_SHIFT_SEARCH = len(zshift_list)
    curr_mse = -1.0
    curr_angle = 0.0
    for z in range(Z_SHIFT_SEARCH):
        if zshift_list is None:
            f_img = f_imgs
            zi = 0.0
        else:
            f_img = f_imgs[z]
            zi = zshift_list[z]
        f_com = get_com(f_img)

        for x in range(N_SHIFT_SEARCH):
            xi = x - MAX_SHIFT_SEARCH + 1
            for y in range(N_SHIFT_SEARCH):
                yi = y - MAX_SHIFT_SEARCH + 1
                tmp_com = f_com + np.array([xi, yi])

                f_pcs, _ = polarTransform.convertToPolarImage(f_img.T, radiusSize=radiusSize, angleSize=angleSize, center=tmp_com)

                angle, min_mse = get_mse_img(g_pcs, f_pcs, angleSize)

                if (min_mse< curr_mse) or (curr_mse==-1.0):
                    curr_mse=min_mse
                    curr_angle = angle
                    curr_com = tmp_com
                    curr_zshift = zi
    theta = curr_angle
    mat = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    g_com_rot = np.dot(mat.T, g_com - imageSize//2) + imageSize//2
    shift_est = np.zeros(3)
    shift_est[:2] = curr_com-g_com_rot
    shift_est[2] = curr_zshift

    return np.rad2deg(theta), shift_est, curr_mse

def q_estimate(psim, pexp, dq, r_matrix, q0, mask=None):
    nmodes = dq.shape[0]
    size = psim.shape[0]
    dy = (psim - pexp)
    if mask is not None:
        dy*= mask
        dq*= mask

    dqt_line = dq.reshape(nmodes, size**2)
    dy_line = dy.reshape(size**2).T
    dqtdq = np.dot(dqt_line, dqt_line.T)

    dq2_inv = np.linalg.inv(dqtdq + r_matrix)

    dqtdy =  np.dot(dqt_line, dy_line) + np.dot(r_matrix, q0)
    return - np.dot( dq2_inv,dqtdy.T)
