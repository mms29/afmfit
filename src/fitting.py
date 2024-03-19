import numpy as np
import time
import polarTransform
from numba import njit
from src.io import PDB
from src.utils import generate_euler_matrix_deg, matrix2eulerAngles, get_points_from_angles, translate, get_angular_distance, get_sphere
from src.simulator import ImageLibrary
import matplotlib.pyplot as plt
from skimage.transform import rotate
from multiprocessing import Pool
import pickle
from src.viewer import FittingPlotter, viewAFMfitting
from warnings import warn
import tqdm

class ProjMatch:
    def __init__(self, img, simulator, pdb):
        self.img = img
        self.simulator = simulator
        self.pdb = pdb
        self.size = simulator.size
        self.vsize = simulator.vsize

        self.angles = None
        self.shifts = None
        self.mse = None
        self.best_angle = None
        self.best_shift = None
        self.best_mse = None
        self.best_fitted_img = None

    def run(self, library,  verbose=True):
        img_exp = self.img
        img_exp_norm = np.sum(np.square(img_exp))
        nimgs = library.nimgs

        # outputs
        mse =    np.zeros((nimgs))
        shifts = np.zeros((nimgs, 3))

        for i in tqdm.tqdm(range(nimgs), desc="Projection Matching"):
            # Rotational Translational XY matching
            _x, _y, _c = trans_match(library.imgs[i],  img_exp)

            mse[i] = library.norm2[i] + img_exp_norm -2*_c
            shifts[i,0] =_x *  self.vsize
            shifts[i,1] =_y *  self.vsize
            shifts[i,2] =library.z_shifts[i]

        self.shifts = shifts
        self.mse = mse
        self.angles = library.angles

        minmse = np.argmin(mse)
        self.best_angle = self.angles[minmse]
        self.best_shift = shifts[minmse]
        fit = self.pdb.copy()
        fit.rotate(self.best_angle)
        fit.translate(self.best_shift)

        img = self.simulator.pdb2afm(fit, 0.0)
        best_mse_calc = mse[minmse]
        self.best_mse = np.sum(np.square( img - img_exp))
        self.best_fitted_img = img

        if verbose:
            print("DONE")
            print("Calculated MSE : %f"%np.sqrt(best_mse_calc))
            print("Actual MSE : %f"%np.sqrt(self.best_mse))

    def show(self, library, n_plot=3):
        fig, ax = plt.subplots(n_plot, 5, figsize=(20, 15))

        minmse = np.argsort(self.mse)
        for i in range(n_plot):
            pest = library.imgs[minmse[i]]
            pest = translate(pest, self.shifts[minmse[i], :2] / self.simulator.vsize)
            mse = self.mse[minmse[i]]
            ax[i, 0].imshow(pest.T, cmap="afmhot", origin="lower")
            ax[i, 1].imshow(self.img.T, cmap="afmhot", origin="lower")
            ax[i, 2].imshow(np.abs(self.img.T - pest.T), cmap="jet", origin="lower")
            ax[i, 3].plot(pest[self.size // 2], color="tab:red")
            ax[i, 3].plot(self.img[self.size // 2], color="tab:blue")
            ax[i, 4].plot(pest[:, self.size // 2], color="tab:red")
            ax[i, 4].plot(self.img[:, self.size // 2], color="tab:blue")
            ax[i, 0].set_title('Fitted [%.2f]' % (np.sqrt(mse)))
            ax[i, 1].set_title('Input')
            ax[i, 2].set_title('Diff')

    def show_sphere(self, angles, nplot=3):
        ax = plt.figure().add_subplot(111, projection='3d')
        all_dir = get_points_from_angles(get_sphere(10))
        ax.scatter(all_dir[:, 0],
                   all_dir[:, 1],
                   all_dir[:, 2], c="grey", s=1)
        directions = get_points_from_angles(angles)
        im = ax.scatter(directions[:, 0],
                        directions[:, 1],
                        directions[:, 2], c=self.mse, cmap="jet")
        for i in range(nplot):
            min_mse = np.argsort(self.mse)[i]
            ax.scatter(directions[min_mse, 0],
                       directions[min_mse, 1],
                       directions[min_mse, 2], c="grey", s=200)
        plt.colorbar(im)
        plt.show()


class AFMFittingSet:
    def __init__(self, pdb, imglib, simulator, nma, target_pdb=None):
        self.pdb = pdb
        self.imglib = imglib
        self.simulator = simulator
        self.nma = nma
        if target_pdb is None:
            target_pdb=[pdb for i in range(imglib.nimg)]
        self.target_pdb = target_pdb
        self.init_library=None

        self.fitlib=None
        self.best_coords=None
        self.best_eta=None
        self.best_angle=None
        self.best_shift=None

        self.mse = None
        self.rmsd = None
        self.dcd = None

    def fit_flexible_rigid_pool(self, n_cpu, outPrefix, n_iter, gamma, gamma_rigid,
                                angular_dist, near_angle_cutoff, n_views, mask=None,
                                plot=True, verbose=False, zshift_range=None, zshift_points=None):

        nimg = self.imglib.nimg
        work_data = []
        self.init_library = self.simulator.get_projection_library(pdb=self.pdb, angular_dist=angular_dist[0], verbose=True,
                                                        init_zshift=None,
                                                        zshift_range=np.linspace(-zshift_range[0], zshift_range[0],
                                                                                 zshift_points[0]))
        for i in range(nimg):
            afmFitting = AFMFitting(pdb=self.pdb, img=self.imglib.imgs[i], simulator=self.simulator, nma=self.nma,
                                    target_pdb=self.target_pdb[i])
            work_data.append([
                i, "%s_%s.pkl" % (outPrefix, str(i).zfill(6)), afmFitting,
                n_iter, gamma, gamma_rigid, angular_dist, near_angle_cutoff, n_views, mask,
                plot, verbose, zshift_range, zshift_points
            ])

        self.pool_handle(n_cpu, work_data)

    def fit_rigid_pool(self, n_cpu, outPrefix,
                       angular_dist, near_angle_cutoff,
                       plot=True, verbose=False, zshift_range=None, zshift_points=None):

        nimg = self.imglib.nimg
        work_data = []
        self.init_library = self.simulator.get_projection_library(pdb=self.pdb, angular_dist=angular_dist[0], verbose=True,
                                                        init_zshift=None,
                                                        zshift_range=np.linspace(-zshift_range[0], zshift_range[0],
                                                                                 zshift_points[0]))
        for i in range(nimg):
            afmFitting = AFMFitting(pdb=self.pdb, img=self.imglib.imgs[i], simulator=self.simulator, nma=self.nma,
                                    target_pdb=self.target_pdb[i])
            work_data.append([
                i, "%s_%s.pkl" % (outPrefix, str(i).zfill(6)), afmFitting,
                angular_dist, near_angle_cutoff,
                plot, verbose, zshift_range, zshift_points
            ])

        print(len(work_data))

        self.pool_handle_rigid(n_cpu, work_data)

    @classmethod
    def from_fitted_files(cls, files):
        n_files = len(files)
        imgs =        []
        fitted_imgs = []
        coords =      []
        eta =         []
        angle =       []
        shift =       []
        mse =         []
        rmsd=         []
        dcd=          []
        for i in range(n_files):
            fit =  AFMFitting.load(files[i])
            imgs.append(fit.img)
            fitted_imgs.append(fit.best_fitted_img)
            coords.append(fit.best_coords)
            eta.append(fit.best_eta)
            angle.append(fit.best_angle)
            shift.append(fit.best_shift)
            mse.append(fit.mse)
            rmsd.append(fit.rmsd)
            dcd.append(fit.dcd)
            pdb = fit.pdb
            simulator = fit.simulator
            nma = fit.nma
            target_pdb = fit.target_pdb
            del fit

        print(fitted_imgs)
        print(len(fitted_imgs))

        imglib = ImageLibrary(np.array(imgs))
        fitSet = AFMFittingSet(pdb, imglib, simulator, nma, target_pdb)

        fitSet.fitlib     = ImageLibrary(np.array(fitted_imgs))
        fitSet.best_coords= np.array(coords)
        fitSet.best_eta   = np.array(eta)
        fitSet.best_angle = np.array(angle)
        fitSet.best_shift = np.array(shift)
        fitSet.mse        = np.array(mse)
        fitSet.rmsd       = np.array(rmsd)
        fitSet.dcd        = np.array(dcd)
        return fitSet

    def show(self):
        viewAFMfitting(self)


    def run_work(self, work_data):
        rank = work_data[0]
        outfile = work_data[1]
        fitter = work_data[2]

        # print("Running work %i ..."%(rank))
        fitter.fit_nma_rotations(*(work_data[3:]+[self.init_library]))
        fitter.dump(outfile)
        # print("Done work %i ..."%(rank))


    def run_work_rigid(self, work_data):
        rank = work_data[0]
        outfile = work_data[1]
        fitter = work_data[2]

        # print("Running work %i ..."%(rank))
        fitter.fit_rotations(*(work_data[3:] + [self.init_library]))
        fitter.dump(outfile)
        # print("Done work %i ..."%(rank))


    def pool_handle(self, n_cpu, work_data):
        p = Pool(n_cpu)
        for _ in tqdm.tqdm(p.imap_unordered(self.run_work, work_data), total=len(work_data)):
            pass
        return p
    def pool_handle_rigid(self, n_cpu, work_data):
        p = Pool(n_cpu)
        for _ in tqdm.tqdm(p.imap_unordered(self.run_work_rigid, work_data), total=len(work_data)):
            pass
        return p


class AFMFitting:
    def __init__(self, pdb, img, simulator, nma, target_pdb=None):
        self.pdb = pdb
        self.img = img
        self.simulator = simulator
        self.nma = nma
        self.plotter = FittingPlotter()
        if target_pdb is None:
            self.target_pdb = pdb
        else:
            self.target_pdb= target_pdb

        # simulation monitoring
        self.dcd = None
        self.mse = None
        self.rmsd = None
        self.eta = None
        self.fitted_imgs = None
        self.projMatch = None

        # TODO
        # self.best_projMatch = None
        self.best_mse= None
        self.best_rmsd = None
        self.best_eta=None
        self.best_fitted_img=None
        self.best_coords = None
        self.best_angle = None
        self.best_shift = None

        # outs

    def dump(self, file):
        with open(file, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file):
        with open(file, "rb") as f:
            return pickle.load(f)

    def show(self):
        self.plotter.start(imgf=self.fitted_imgs[-1], imge=self.img, mse=self.mse, rmsd=self.rmsd)

    def fit_nma(self, n_iter, gamma=50, gamma_rigid=5, mask=None, verbose=True, plot=False, q_init=None, plotter=None,
                zshift = None):
        dtimetot = time.time()

        # Inputs
        if zshift is None:
            zshift=0.0
        nma=self.nma
        pdb=self.pdb
        size = self.simulator.size
        pexp = self.img

        # define output arrays
        self.dcd = np.zeros((n_iter+1, pdb.n_atoms, 3))
        self.mse = np.zeros(n_iter+1)
        self.rmsd =np.zeros(n_iter+1)
        self.fitted_imgs =np.zeros((n_iter+1, size, size))
        self.eta = np.zeros((n_iter + 1, nma.nmodes_total))
        if q_init is not None:
            self.eta[0] = q_init

        # init state
        fitted = nma.applySpiralTransformation(eta=self.eta[0], pdb=pdb)
        psim = self.simulator.pdb2afm(fitted, zshift=zshift)
        self.mse[0]=np.linalg.norm(psim - pexp)**2
        self.rmsd[0]=fitted.getRMSD(self.target_pdb, align=True)
        self.dcd[0] = fitted.coords
        self.fitted_imgs[0] = psim

        # Define regularization
        regul = np.zeros( nma.nmodes_total)
        regul[:6] = 1/(gamma_rigid**2)
        regul[6:] = 1/(gamma**2)
        linear_modes_line =  nma.linear_modes.reshape(nma.nmodes_total, nma.natoms * 3)
        r_matrix = np.dot(linear_modes_line, linear_modes_line.T) * regul

        if plot:
            if plotter is not None:
                plotter.update_imgs(psim, pexp)
                plotter.draw()
            else:
                self.plotter.start(psim, pexp, self.mse, self.rmsd, interactive=True)

        for i in range(n_iter):
            # Compute the gradient
            dq = self.simulator.pdb2afm_grad(pdb=fitted,nma=nma, psim=psim, zshift=zshift)

            # Compute the new NMA amplitudes
            q_est = q_estimate(psim, pexp, dq,r_matrix=r_matrix, q0=self.eta[i], mask=mask)
            self.eta[i + 1] = q_est + self.eta[i]

            # Apply NMA amplitudes to the PDB
            fitted = nma.applySpiralTransformation(eta=self.eta[i+1], pdb=pdb)

            # Simulate the new image
            psim = self.simulator.pdb2afm(fitted, zshift=zshift)

            # Update outputs
            self.dcd[i+1] = fitted.coords
            self.mse[i+1] = np.linalg.norm(psim - pexp)**2
            self.rmsd[i+1]= fitted.getRMSD(self.target_pdb, align=True)
            self.fitted_imgs[i+1]=psim


            norm = np.array([np.linalg.norm(self.eta[i+1]-self.eta[q]) for q in range(i)])
            if any(norm<  1.0):
                for q in range(i, n_iter):
                    self.dcd[q + 1] = self.dcd[i+1]
                    self.mse[q + 1] = self.mse[i+1]
                    self.rmsd[q + 1] = self.rmsd[i+1]
                    self.fitted_imgs[q + 1] = self.fitted_imgs[i+1]
                break

            if verbose:
                print("Iter %i = %.2f ; %.2f" % (i, self.mse[i+1], self.rmsd[i+1]))
                print(self.eta[i + 1])

            if plot:
                if plotter is not None:
                    plotter.update_imgs(psim, pexp)
                    plotter.draw()
                else:
                    self.plotter.update(psim, pexp, self.mse, self.rmsd)

        best_idx= np.argmin(self.mse)
        self.best_mse= self.mse[best_idx]
        self.best_rmsd = self.rmsd[best_idx]
        self.best_eta= self.eta[best_idx]
        self.best_fitted_img=self.fitted_imgs[best_idx]
        self.best_coords = self.dcd[best_idx]

        R, t = PDB.alignCoords( self.best_coords, self.pdb.coords)
        self.best_angle = matrix2eulerAngles(R.T)
        self.best_shift = t

        if verbose:
            print("\t TIME_TOTAL   => %.4f s" % (time.time() - dtimetot))



    def fit_nma_rotations(self, n_iter, gamma, gamma_rigid,
                angular_dist, near_angle_cutoff, n_views, mask =None,
                plot = True, verbose = False, zshift_range=0.0, zshift_points=1, init_library=None):

        # Inputs
        maxiter = len(angular_dist)
        psim = self.simulator.pdb2afm(self.pdb)
        pexp = self.img
        pdb = self.pdb
        fitted = pdb.copy()
        curr_mse = np.linalg.norm(psim- pexp)**2
        curr_eta = np.zeros( self.nma.nmodes_total)
        best_zshift=None
        best_angle=np.zeros(3)
        best_shift=np.zeros(3)
        near_angle=None

        # Output arrays
        self.mse = np.array([curr_mse])
        self.rmsd = np.array([pdb.getRMSD(self.target_pdb, align=True)])
        self.dcd = np.array([pdb.coords])
        self.eta = np.array([curr_eta])
        self.fitted_imgs = np.array([psim])
        self.projMatch = []

        if plot:
            self.plotter.start(psim, pexp, self.mse , self.rmsd, interactive=True)

        for ii in range(maxiter):
            if verbose:
                print("Iteration : %i"%ii)
            zshift_list = np.linspace(-zshift_range[ii], zshift_range[ii], zshift_points[ii])

            # get image library
            if ii == 0 or near_angle_cutoff[ii] == -1:
                if ii == 0 and init_library is not None:
                    image_library = init_library
                else:
                    image_library= self.simulator.get_projection_library(fitted,
                                            zshift_range=zshift_list, angular_dist=angular_dist[ii], verbose=verbose)
            else:
                image_library= self.simulator.get_projection_library(fitted, near_angle=near_angle, init_zshift=best_zshift,
                    near_angle_cutoff=near_angle_cutoff[ii], zshift_range=zshift_list, angular_dist=angular_dist[ii], verbose=verbose)

            # performs projection matching
            projMatch = ProjMatch(library=image_library, img = pexp, simulator=self.simulator, pdb=fitted)
            projMatch.run(angular_dist=angular_dist[ii],verbose=verbose)
            self.projMatch.append(projMatch)

            # get ordered array of best views
            best_views_idx = np.argsort(projMatch.mse)
            if len(best_views_idx)< n_views[ii]:
                n_best = len(best_views_idx)
                warn("Could not get the requested number of views %i /%i"%(n_best,n_views[ii]))
            else:
                best_views_idx = best_views_idx[:n_views[ii]]
                n_best = n_views[ii]

            # loop over the best views and run NMA fitting
            nmafits = []
            for view in range(n_best):
                angle_view = projMatch.angles[best_views_idx[view]]
                shift_view = projMatch.shifts[best_views_idx[view]]

                # rotate NMA
                rot_nma = self.nma.transform(angle=angle_view, shift=shift_view)
                candidate = rot_nma.pdb

                # Fit NMA
                nmafit = AFMFitting(pdb=candidate, img=self.img,
                           simulator=self.simulator, nma=rot_nma, target_pdb=self.target_pdb)
                nmafit.fit_nma(n_iter=n_iter, gamma=gamma, gamma_rigid=gamma_rigid, mask=mask, verbose=verbose, plot=plot,
                                  plotter=self.plotter,  q_init=curr_eta, zshift = 0.0)
                nmafits.append(nmafit)

            # get best NMA fit
            best_fit_idx = np.argmin([np.min(nmafit.mse) for nmafit in nmafits])
            best_nmafit = nmafits[best_fit_idx]
            best_view_idx = best_views_idx[best_fit_idx]
            best_iter_idx = np.argmin(best_nmafit.mse)
            best_mse = best_nmafit.mse[best_iter_idx]

            # update alignemnts
            if best_mse< curr_mse:
                curr_eta = best_nmafit.eta[best_iter_idx]
                near_angle = image_library.angles[best_view_idx]
                best_angle = projMatch.angles[best_view_idx]
                best_shift = projMatch.shifts[best_view_idx]
                best_zshift = best_shift[2]
                curr_mse = best_mse
            else:
                warn("Could not improve the fitting")
                if ii == 0:
                    return

            # update outputs
            self.mse = np.concatenate((self.mse, best_nmafit.mse))
            self.rmsd = np.concatenate((self.rmsd, best_nmafit.rmsd))
            self.dcd = np.concatenate((self.dcd, best_nmafit.dcd))
            self.fitted_imgs = np.concatenate((self.fitted_imgs, best_nmafit.fitted_imgs))
            self.eta = np.concatenate((self.eta, best_nmafit.eta))

            fitted = self.nma.applySpiralTransformation(eta=curr_eta, pdb=pdb)
            fitted_rot = fitted.copy()
            fitted_rot.rotate(best_angle)
            fitted_rot.translate(best_shift)
            psim = self.simulator.pdb2afm(fitted_rot, zshift= 0.0)
            self.fitted_imgs = np.concatenate((self.fitted_imgs,np.array([psim])))

            if plot:
                self.plotter.update(psim, pexp, self.mse, self.rmsd,
                                    mse_a = [f.mse for f in nmafits],
                                    rmsd_a = [f.rmsd for f in nmafits])

        best_idx= np.argmin(self.mse)
        self.best_mse= self.mse[best_idx]
        self.best_rmsd = self.rmsd[best_idx]
        self.best_eta= self.eta[best_idx]
        self.best_fitted_img=self.fitted_imgs[best_idx]
        self.best_coords = self.dcd[best_idx]

        R, t = PDB.alignCoords( self.best_coords, self.pdb.coords)
        self.best_angle = matrix2eulerAngles(R.T)
        self.best_shift = t


    def fit_nma_rotations(self, n_iter, gamma, gamma_rigid,
                angular_dist, near_angle_cutoff, n_views, mask =None,
                plot = True, verbose = False, zshift_range=0.0, zshift_points=1, init_library=None):

        # Inputs
        maxiter = len(angular_dist)
        psim = self.simulator.pdb2afm(self.pdb)
        pexp = self.img
        pdb = self.pdb
        fitted = pdb.copy()
        curr_mse = np.linalg.norm(psim- pexp)**2
        curr_eta = np.zeros( self.nma.nmodes_total)
        best_zshift=None
        best_angle=np.zeros(3)
        best_shift=np.zeros(3)
        near_angle=None

        # Output arrays
        self.mse = np.array([curr_mse])
        self.rmsd = np.array([pdb.getRMSD(self.target_pdb, align=True)])
        self.dcd = np.array([pdb.coords])
        self.eta = np.array([curr_eta])
        self.fitted_imgs = np.array([psim])
        self.projMatch = []

        if plot:
            self.plotter.start(psim, pexp, self.mse , self.rmsd, interactive=True)

        for ii in range(maxiter):
            if verbose:
                print("Iteration : %i"%ii)
            zshift_list = np.linspace(-zshift_range[ii], zshift_range[ii], zshift_points[ii])

            # get image library
            if ii == 0 or near_angle_cutoff[ii] == -1:
                if ii == 0 and init_library is not None:
                    image_library = init_library
                else:
                    image_library= self.simulator.get_projection_library(fitted,
                                            zshift_range=zshift_list, angular_dist=angular_dist[ii], verbose=verbose)
            else:
                image_library= self.simulator.get_projection_library(fitted, near_angle=near_angle, init_zshift=best_zshift,
                    near_angle_cutoff=near_angle_cutoff[ii], zshift_range=zshift_list, angular_dist=angular_dist[ii], verbose=verbose)

            # performs projection matching
            projMatch = ProjMatch(library=image_library, img = pexp, simulator=self.simulator, pdb=fitted)
            projMatch.run(angular_dist=angular_dist[ii],verbose=verbose)
            self.projMatch.append(projMatch)

            # get ordered array of best views
            best_views_idx = np.argsort(projMatch.mse)
            if len(best_views_idx)< n_views[ii]:
                n_best = len(best_views_idx)
                warn("Could not get the requested number of views %i /%i"%(n_best,n_views[ii]))
            else:
                best_views_idx = best_views_idx[:n_views[ii]]
                n_best = n_views[ii]

            # loop over the best views and run NMA fitting
            nmafits = []
            for view in range(n_best):
                angle_view = projMatch.angles[best_views_idx[view]]
                shift_view = projMatch.shifts[best_views_idx[view]]

                # rotate NMA
                rot_nma = self.nma.transform(angle=angle_view, shift=shift_view)
                candidate = rot_nma.pdb

                # Fit NMA
                nmafit = AFMFitting(pdb=candidate, img=self.img,
                           simulator=self.simulator, nma=rot_nma, target_pdb=self.target_pdb)
                nmafit.fit_nma(n_iter=n_iter, gamma=gamma, gamma_rigid=gamma_rigid, mask=mask, verbose=verbose, plot=plot,
                                  plotter=self.plotter,  q_init=curr_eta, zshift = 0.0)
                nmafits.append(nmafit)

            # get best NMA fit
            best_fit_idx = np.argmin([np.min(nmafit.mse) for nmafit in nmafits])
            best_nmafit = nmafits[best_fit_idx]
            best_view_idx = best_views_idx[best_fit_idx]
            best_iter_idx = np.argmin(best_nmafit.mse)
            best_mse = best_nmafit.mse[best_iter_idx]

            # update alignemnts
            if best_mse< curr_mse:
                curr_eta = best_nmafit.eta[best_iter_idx]
                near_angle = image_library.angles[best_view_idx]
                best_angle = projMatch.angles[best_view_idx]
                best_shift = projMatch.shifts[best_view_idx]
                best_zshift = best_shift[2]
                curr_mse = best_mse
            else:
                warn("Could not improve the fitting")
                if ii == 0:
                    return

            # update outputs
            self.mse = np.concatenate((self.mse, best_nmafit.mse))
            self.rmsd = np.concatenate((self.rmsd, best_nmafit.rmsd))
            self.dcd = np.concatenate((self.dcd, best_nmafit.dcd))
            self.fitted_imgs = np.concatenate((self.fitted_imgs, best_nmafit.fitted_imgs))
            self.eta = np.concatenate((self.eta, best_nmafit.eta))

            fitted = self.nma.applySpiralTransformation(eta=curr_eta, pdb=pdb)
            fitted_rot = fitted.copy()
            fitted_rot.rotate(best_angle)
            fitted_rot.translate(best_shift)
            psim = self.simulator.pdb2afm(fitted_rot, zshift= 0.0)
            self.fitted_imgs = np.concatenate((self.fitted_imgs,np.array([psim])))

            if plot:
                self.plotter.update(psim, pexp, self.mse, self.rmsd,
                                    mse_a = [f.mse for f in nmafits],
                                    rmsd_a = [f.rmsd for f in nmafits])

        best_idx= np.argmin(self.mse)
        self.best_mse= self.mse[best_idx]
        self.best_rmsd = self.rmsd[best_idx]
        self.best_eta= self.eta[best_idx]
        self.best_fitted_img=self.fitted_imgs[best_idx]
        self.best_coords = self.dcd[best_idx]

        R, t = PDB.alignCoords( self.best_coords, self.pdb.coords)
        self.best_angle = matrix2eulerAngles(R.T)
        self.best_shift = t

    def fit_rotations(self, angular_dist, near_angle_cutoff,
                plot = True, verbose = False, zshift_range=0.0, zshift_points=1, init_library=None):

        # Inputs
        maxiter = len(angular_dist)
        psim = self.simulator.pdb2afm(self.pdb)
        pexp = self.img
        best_zshift=None
        near_angle=None

        # Output arrays
        self.mse = np.array([np.linalg.norm(psim- pexp)**2])
        self.rmsd = np.array([0.0])
        self.fitted_imgs = np.array([psim])
        self.projMatch = []

        if plot:
            self.plotter.start(psim, pexp, self.mse , self.rmsd, interactive=True)

        for ii in range(maxiter):
            if verbose:
                print("Iteration : %i"%ii)
            zshift_list = np.linspace(-zshift_range[ii], zshift_range[ii], zshift_points[ii])

            # get image library
            if ii == 0 or near_angle_cutoff[ii] == -1:
                if ii == 0 and init_library is not None:
                    image_library = init_library
                else:
                    image_library= self.simulator.get_projection_library(self.pdb,
                                            zshift_range=zshift_list, angular_dist=angular_dist[ii], verbose=verbose)
            else:
                image_library= self.simulator.get_projection_library(self.pdb, near_angle=near_angle, init_zshift=best_zshift,
                    near_angle_cutoff=near_angle_cutoff[ii], zshift_range=zshift_list, angular_dist=angular_dist[ii], verbose=verbose)

            # performs projection matching
            projMatch = ProjMatch(library=image_library, img = pexp, simulator=self.simulator, pdb=self.pdb)
            projMatch.run(angular_dist=angular_dist[ii],verbose=verbose)
            self.projMatch.append(projMatch)

            best_zshift = projMatch.best_shift[2]
            near_angle = projMatch.best_angle_sphere

            # update outputs
            self.mse = np.concatenate((self.mse, np.array([projMatch.best_mse])))
            self.rmsd = np.concatenate((self.rmsd, np.array([0.0])))
            self.fitted_imgs = np.concatenate((self.fitted_imgs, np.array([projMatch.best_fitted_img])))

            if plot:
                self.plotter.update(projMatch.best_fitted_img, pexp, self.mse, self.rmsd)


        best_idx= np.argmin(self.mse)
        self.best_mse= self.mse[best_idx]
        self.best_rmsd = self.rmsd[best_idx]
        self.best_fitted_img=self.fitted_imgs[best_idx]
        self.best_coords = self.pdb.coords
        self.best_angle = self.projMatch[best_idx-1].best_angle
        self.best_shift = self.projMatch[best_idx-1].best_shift


# @njit
# def corr_sum_njit(f_pcs_fft, g_pcs_fft, angleSize, radiusSize):
#     corr_fourier = np.zeros((angleSize), dtype=np.complex128)
#     for r in range(radiusSize):
#         corr_fourier += r * g_pcs_fft[:, r] * np.conjugate(f_pcs_fft[:, r])
#     return corr_fourier


# def get_corr_fourier(f_pcs_fft, g_pcs_fft, angleSize, radiusSize):
#     corr_fourier = corr_sum_njit(f_pcs_fft, g_pcs_fft, angleSize, radiusSize)
#     corr = np.fft.ifftn(corr_fourier).real
#
#     argmax_corr = np.argmax(corr)
#     max_corr = corr[argmax_corr]
#     return  (argmax_corr*np.pi*2)/angleSize, max_corr

def get_corr_fourier(g_pcs_fft, f_pcs_fft, angleSize):
    corr_fourier = np.sum(g_pcs_fft * np.conjugate(f_pcs_fft), axis=1)
    corr = np.fft.ifftn(corr_fourier).real
    argmax_corr = np.argmax(corr)
    max_corr = corr[argmax_corr]
    return  (argmax_corr*np.pi*2)/angleSize, max_corr
def get_mse_img(g_pcs, f_pcs,angleSize):
    mse = np.zeros(angleSize)
    for i in range(angleSize):
        mse[i] = np.sum(np.square(g_pcs - np.roll(f_pcs, i, axis=0) ))
    min_mse = np.argmin(mse)
    return (min_mse*np.pi*2)/angleSize, mse[min_mse]


def trans_match(img1, img2):
    size = img1.shape[0]
    img1_ft = np.fft.fftn(img1)
    img2_ft = np.fft.fftn(img2)
    corr  = np.fft.ifftn(img1_ft * np.conjugate(img2_ft)).real
    shiftx = np.argmax(corr.sum(axis=1))
    shifty = np.argmax(corr.sum(axis=0))
    if shiftx>= size//2:
        shiftx-= size
    if shifty>= size//2:
        shifty-= size
    return -shiftx, -shifty, np.max(corr)

def rot_trans_match(img1, img2, angleSize, img2_norm2=None):
    if img2_norm2 is None:
        img2_norm2 = np.sum(np.square(img2))
    mse = np.zeros(angleSize)
    shiftx = np.zeros(angleSize)
    shifty = np.zeros(angleSize)
    for i in range(angleSize):
        ang = (360/angleSize)*i
        rot_img = rotate(img1, ang)
        rot_img_norm2 = np.sum(np.square(rot_img))
        _x, _y, _c = trans_match(rot_img, img2)
        mse[i] = rot_img_norm2 + img2_norm2 -2*_c
        shiftx[i] = _x
        shifty[i] = _y

    minmse =  np.argmin(mse)
    angle = np.rad2deg(minmse*np.pi*2/angleSize)
    shift = np.array([shiftx[minmse], shifty[minmse]])
    return angle, shift, mse[minmse]

def rot_trans_match_full(imgs1, img2, imgs1_norm2, img2_norm2=None):
    nimgs = imgs1.shape[0]
    if img2_norm2 is None:
        img2_norm2 = np.sum(np.square(img2))
    mse = np.zeros(nimgs)
    shiftx = np.zeros(nimgs)
    shifty = np.zeros(nimgs)
    for i in range(nimgs):
        _x, _y, _c = trans_match(imgs1[i], img2)
        mse[i] = imgs1_norm2[i] + img2_norm2 -2*_c
        shiftx[i] = _x
        shifty[i] = _y

    min_mse = np.argmin(mse)

    shift = np.array([shiftx[min_mse], shifty[min_mse]])

    return min_mse, mse[min_mse], shift

# def rot_trans_z_match_full(imgs1, img2, img2_norm2, zshift_range):
#     #inputs
#     ZERO_CUTOFF = 5.0  # Angstrom
#     n_zviews = imgs1.shape[0]
#     n_zshifts = len(zshift_range)
#
#     # create arrays
#     mse = np.zeros((n_zviews, n_zshifts))
#     shift = np.zeros((n_zviews, n_zshifts,2))
#     for i in range(n_zviews):
#         for z in range(n_zshifts):
#             img_zshift = imgs1[i].copy()
#             img_zshift[img_zshift >= ZERO_CUTOFF] += zshift_range[z]
#             if zshift_range[z] < 0:
#                 img_zshift[img_zshift < 0.0] = 0.0
#             img_zshift_norm2 = np.sum(np.square(img_zshift))
#
#             _x, _y, _c = trans_match(img_zshift, img2)
#             mse[i, z] = img_zshift_norm2 + img2_norm2 -2*_c
#             shift[i,z, 0] = _x
#             shift[i,z, 1] = _y
#
#     return mse, shift

# def rot_trans_z_match(imgs, img_exp, angleSize):
#     NZSHIFT = len(imgs)
#     angle = np.zeros(NZSHIFT)
#     shiftxy = np.zeros((NZSHIFT,2))
#     mse = np.zeros(NZSHIFT)
#     for i in range(NZSHIFT):
#         _a, _xy, _mse = rot_trans_match(imgs[i], img_exp, angleSize=angleSize)
#         angle[i] = _a
#         shiftxy[i] = _xy
#         mse[i] = _mse
#     minmse = np.argmin(mse)
#
#     return angle[minmse], shiftxy[minmse],  minmse, mse[minmse]


# def rotational_matching(g_img, f_imgs, imageSize, angleSize, radiusSize, zshift_list=None, MAX_SHIFT_SEARCH = 5):
#     N_SHIFT_SEARCH = MAX_SHIFT_SEARCH*2 - 1
#     if zshift_list is None:
#         Z_SHIFT_SEARCH = 1
#     else:
#         Z_SHIFT_SEARCH = len(zshift_list)
#     curr_mse = -1.0
#     corr_matrix = np.zeros((Z_SHIFT_SEARCH, N_SHIFT_SEARCH, N_SHIFT_SEARCH))
#     g_pcs_stk = np.zeros((Z_SHIFT_SEARCH, N_SHIFT_SEARCH, N_SHIFT_SEARCH, angleSize, radiusSize))
#     angle_est = np.zeros(3)
#     shift_est = np.zeros(3)
#
#
#     f_com = np.array([imageSize/2, imageSize/2])
#     g_com = get_com(g_img)
#
#     f_pcs, _ = polarTransform.convertToPolarImage(f_img.T, radiusSize=radiusSize, angleSize=angleSize,
#                                                   center=f_com)
#     f_pcs_fft = np.fft.fft(f_pcs, axis=0)
#     f_sum = np.sum(np.square(f_pcs))
#
#     for x in range(N_SHIFT_SEARCH):
#         xi = x - MAX_SHIFT_SEARCH + 1
#         for y in range(N_SHIFT_SEARCH):
#             yi = y - MAX_SHIFT_SEARCH + 1
#             tmp_com = g_com + np.array([xi, yi])
#
#             for z in range(Z_SHIFT_SEARCH):
#                 if zshift_list is None:
#                     f_img = f_imgs
#                     zi = 0.0
#                 else:
#                     f_img = f_imgs[z]
#                     zi = zshift_list[z]
#
#                 g_pcs, _ = polarTransform.convertToPolarImage(g_img.T, radiusSize=radiusSize, angleSize=angleSize,
#                                                               center=tmp_com)
#                 g_pcs_fft = np.fft.fft(g_pcs, axis=0)
#                 g_sum = np.sum(np.square(g_pcs))
#                 g_pcs_stk [z,x,y] = g_pcs
#
#                 angle, corr = get_corr_fourier(g_pcs_fft= g_pcs_fft, f_pcs_fft=f_pcs_fft, angleSize=angleSize)
#                 min_mse = f_sum + g_sum + -2 * corr
#                 corr_matrix[z,x,y] = min_mse
#
#                 if (min_mse< curr_mse) or (curr_mse==-1.0):
#                     curr_mse=min_mse
#                     angle_est = np.rad2deg(angle)
#                     shift_est[:2] = tmp_com-f_com
#                     shift_est[2] = -zi
#
#     return angle_est, shift_est, curr_mse, corr_matrix, g_pcs_stk

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

