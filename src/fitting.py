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
from multiprocessing.sharedctypes import RawArray
from numpy import frombuffer

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

        for i in tqdm.tqdm(range(nimgs), desc="Projection Matching", disable=not verbose):
            # Rotational Translational XY matching
            _x, _y, _c = self.trans_match(library.get_img(i),  img_exp)

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

    @classmethod
    def trans_match(cls, img1, img2):
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

class NMAFit:
    def __init__(self):
        self.best_mse = None
        self.best_rmsd = None
        self.mse = None
        self.rmsd = None
        self.best_eta = None
        self.best_fitted_img = None
        self.best_coords  = None
        self.best_angle = None
        self.best_shift = None

    def fit(self, img, nma, simulator, target_pdb=None, zshift=None,
            n_iter=10, gamma=50, gamma_rigid=5, verbose=True, plot=False, q_init=None, init_plotter=None):
        dtimetot = time.time()

        # Inputs
        pdb = nma.pdb
        pexp = img
        if target_pdb is None:
            target_pdb = pdb
        if zshift is None:
            zshift=0.0

        # define output arrays
        dcd = np.zeros((n_iter+1, pdb.n_atoms, 3), dtype=np.float32)
        mse = np.zeros(n_iter+1)
        rmsd =np.zeros(n_iter+1)
        eta = np.zeros((n_iter + 1, nma.nmodes_total))
        fitted_imgs =np.zeros((n_iter+1, simulator.size, simulator.size), dtype=np.float32)
        if q_init is not None:
            eta[0] = q_init

        # init state
        fitted = nma.applySpiralTransformation(eta=eta[0], pdb=pdb)
        psim = simulator.pdb2afm(fitted, zshift=zshift)
        mse[0]=np.linalg.norm(psim - pexp)**2
        rmsd[0]=fitted.getRMSD(target_pdb, align=True)
        dcd[0] = fitted.coords
        fitted_imgs[0] = psim

        # Define regularization
        regul = np.zeros( nma.nmodes_total)
        regul[:6] = 1/(gamma_rigid**2)
        regul[6:] = 1/(gamma**2)
        linear_modes_line =  nma.linear_modes.reshape(nma.nmodes_total, nma.natoms * 3)
        r_matrix = np.dot(linear_modes_line, linear_modes_line.T) * regul

        if plot:
            if init_plotter is not None:
                plotter = init_plotter
                plotter.update_imgs(psim, pexp)
                plotter.draw()
            else:
                plotter = FittingPlotter()
                plotter.start(psim, pexp, mse, rmsd, interactive=True)

        for i in range(n_iter):
            # Compute the gradient
            dq = simulator.pdb2afm_grad(pdb=fitted,nma=nma, psim=psim, zshift=zshift)

            # Compute the new NMA amplitudes
            q_est = self.q_estimate(psim, pexp, dq,r_matrix=r_matrix, q0=eta[i])
            eta[i + 1] = q_est + eta[i]

            # Apply NMA amplitudes to the PDB
            fitted = nma.applySpiralTransformation(eta=eta[i+1], pdb=pdb)

            # Simulate the new image
            psim = simulator.pdb2afm(fitted, zshift=zshift)

            # Update outputs
            dcd[i+1] = fitted.coords
            mse[i+1] = np.linalg.norm(psim - pexp)**2
            rmsd[i+1]= fitted.getRMSD(target_pdb, align=True)
            fitted_imgs[i+1]= psim

            # If the trajectory has converged, stop the iterations
            norm = np.array([np.linalg.norm(eta[i+1]-eta[q]) for q in range(i)])
            if any(norm<  1.0):
                for q in range(i, n_iter):
                    dcd[q + 1] = dcd[i+1]
                    mse[q + 1] = mse[i+1]
                    rmsd[q + 1] = rmsd[i+1]
                    fitted_imgs[q + 1] = fitted_imgs[i+1]
                break

            if verbose:
                print("Iter %i = %.2f ; %.2f" % (i, mse[i+1], rmsd[i+1]))
                print(eta[i + 1])

            if plot:
                if init_plotter is not None:
                    plotter.update_imgs(psim, pexp)
                    plotter.draw()
                else:
                    plotter.update(psim, pexp, mse, rmsd)

        # outputs
        best_idx= np.argmin(mse)
        self.best_mse = mse[best_idx]
        self.best_rmsd= rmsd[best_idx]
        self.mse = mse
        self.rmsd= rmsd

        self.best_eta= eta[best_idx]
        self.best_fitted_img=fitted_imgs[best_idx]
        self.best_coords = dcd[best_idx]

        R, t = PDB.alignCoords(dcd[best_idx], pdb.coords)
        self.best_angle= matrix2eulerAngles(R.T)
        self.best_shift= t

        if verbose:
            print("\t TIME_TOTAL   => %.4f s" % (time.time() - dtimetot))

    def q_estimate(self, psim, pexp, dq, r_matrix, q0, mask=None):
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


class Fitter:
    def __init__(self, pdb, imgs, simulator, target_pdbs=None):
        # inputs
        self.pdb = pdb
        self.imgs = imgs
        self.nimgs = len(imgs)
        self.simulator = simulator
        if target_pdbs is None:
            self.target_pdbs = [pdb for i in range(self.nimgs)]
        else:
            self.target_pdbs= target_pdbs

        #RIGID outputs
        self.rigid_angles=  None
        self.rigid_shifts=  None
        self.rigid_mses = None

        #Flexible outputs
        self.flexible_angles = None
        self.flexible_shifts = None
        self.flexible_eta = None
        self.flexible_mses = None
        self.flexible_rmsds = None
        self.flexible_coords = None

    def dump(self, file):
        with open(file, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file):
        with open(file, "rb") as f:
            return pickle.load(f)

    def init_projmatch_processes(self, library,anglesRawArray,shiftsRawArray, msesRawArray, nimgs, n_best_views):
        global library_global
        global angles_global
        global shifts_global
        global mses_global
        library_global = library

        angles_global = frombuffer(anglesRawArray, dtype=np.float64,
                                   count=len(anglesRawArray)).reshape(nimgs,n_best_views,3)

        shifts_global = frombuffer(shiftsRawArray, dtype=np.float64,
                                   count=len(shiftsRawArray)).reshape(nimgs,n_best_views, 3)
        mses_global = frombuffer(msesRawArray, dtype=np.float64,
                                   count=len(msesRawArray)).reshape(nimgs,n_best_views)


    def run_projmatch_processes(self, workdata):
        rank = workdata[0]
        n_best_views = workdata[3]

        projMatch = ProjMatch(img=self.imgs[rank], simulator=self.simulator, pdb=self.pdb)
        projMatch.run(library=library_global, verbose=False)

        idx_best_views =np.argsort(projMatch.mse)[:n_best_views]
        angles_global[rank,:] = projMatch.angles[idx_best_views]
        shifts_global[rank,:] = projMatch.shifts[idx_best_views]
        mses_global[rank,:] = projMatch.mse[idx_best_views]

    def fit_rigid(self, n_cpu, angular_dist=10.0, verbose=False, zshift_range=None, n_best_views=3):

        #inputs
        if zshift_range is None:
            zshift_range = np.linspace(-20.0,20.0,10)

        # create library
        library = self.simulator.project_library(n_cpu=n_cpu, pdb=self.pdb,  angular_dist=angular_dist, verbose=verbose,
                    zshift_range=zshift_range)

        workdata=[]
        for i in range(self.nimgs):
            workdata.append(
                [i, self.imgs[i], self.nimgs, n_best_views]
            )
        # create output arrays
        anglesRawArray = RawArray("d", self.nimgs*3*n_best_views)
        shiftsRawArray = RawArray("d", self.nimgs*3*n_best_views)
        msesRawArray = RawArray("d", self.nimgs*n_best_views)

        # run
        p = Pool(n_cpu, initializer=self.init_projmatch_processes, initargs=(library, anglesRawArray, shiftsRawArray,msesRawArray, self.nimgs, n_best_views))
        for _ in tqdm.tqdm(p.imap_unordered(self.run_projmatch_processes, workdata), total=len(workdata), desc="Projection Matching"
                           , disable=not verbose):
            pass

        # get arrays
        angles = frombuffer(anglesRawArray, dtype=np.float64,
                                   count=len(anglesRawArray)).reshape(self.nimgs,n_best_views, 3)

        shifts = frombuffer(shiftsRawArray, dtype=np.float64,
                                   count=len(shiftsRawArray)).reshape(self.nimgs,n_best_views, 3)
        mses = frombuffer(msesRawArray, dtype=np.float64,
                                   count=len(msesRawArray)).reshape(self.nimgs,n_best_views)

        self.rigid_angles= angles
        self.rigid_shifts= shifts
        self.rigid_mses= mses

    def fit_flexible(self, n_cpu, nma, angular_dist=10.0, verbose=False, zshift_range=None, n_best_views=3,
                     n_iter=10, gamma=50, gamma_rigid=10, plot=False):
        self.fit_rigid(n_cpu=n_cpu, angular_dist=angular_dist,
                        verbose=verbose, zshift_range=zshift_range, n_best_views=n_best_views)


        msesRawArray   = RawArray("d", self.nimgs * (n_iter+1))
        rmsdsRawArray  = RawArray("d", self.nimgs * (n_iter+1))
        coordsRawArray = RawArray("f", self.nimgs * nma.pdb.n_atoms * 3)
        etaRawArray    = RawArray("d", self.nimgs * nma.nmodes_total)
        anglesRawArray = RawArray("d", self.nimgs*3)
        shiftsRawArray = RawArray("d", self.nimgs*3)

        workdata = []
        for i in range(self.nimgs):
            workdata.append([i])

        p = Pool(n_cpu, initializer=self.init_flexible_processes, initargs=(rmsdsRawArray,anglesRawArray,
                                shiftsRawArray, msesRawArray, coordsRawArray, etaRawArray,
                                nma, n_iter, n_best_views, plot, gamma, gamma_rigid))
        for _ in tqdm.tqdm(p.imap_unordered(self.run_flexible_processes, workdata), total=len(workdata), desc="Flexible Fitting"
                           , disable=not verbose):
            pass

        self.flexible_mses = frombuffer(msesRawArray, dtype=np.float64,
                                   count=len(msesRawArray)).reshape(self.nimgs, (n_iter+1))
        self.flexible_rmsds = frombuffer(rmsdsRawArray, dtype=np.float64,
                                   count=len(rmsdsRawArray)).reshape(self.nimgs, (n_iter+1))
        self.flexible_coords = frombuffer(coordsRawArray, dtype=np.float32,
                                   count=len(coordsRawArray)).reshape(self.nimgs, nma.pdb.n_atoms, 3)
        self.flexible_eta = frombuffer(etaRawArray, dtype=np.float64,
                                   count=len(etaRawArray)).reshape(self.nimgs, nma.nmodes_total)
        self.flexible_angles = frombuffer(anglesRawArray, dtype=np.float64,
                                   count=len(anglesRawArray)).reshape(self.nimgs, 3)
        self.flexible_shifts = frombuffer(shiftsRawArray, dtype=np.float64,
                                   count=len(shiftsRawArray)).reshape(self.nimgs, 3)

        for i in range(self.nimgs):
            R, t = PDB.alignCoords( self.flexible_coords[i], self.pdb.coords)
            self.flexible_angles[i] = matrix2eulerAngles(R.T)
            self.flexible_shifts[i] = t

    def init_flexible_processes(self, rmsdsRawArray,anglesRawArray,shiftsRawArray, msesRawArray, coordsRawArray, etaRawArray,
                                nma, n_iter, n_best_views, plot, gamma, gamma_rigid):
        global eta_global
        global rmsds_global
        global coords_global
        global angles_global
        global shifts_global
        global mses_global
        global n_best_views_global
        global nma_global
        global plot_global
        global n_iter_global
        global gamma_global
        global gamma_rigid_global
        mses_global = frombuffer(msesRawArray, dtype=np.float64,
                                   count=len(msesRawArray)).reshape(self.nimgs, (n_iter+1))
        rmsds_global = frombuffer(rmsdsRawArray, dtype=np.float64,
                                   count=len(rmsdsRawArray)).reshape(self.nimgs, (n_iter+1))
        coords_global = frombuffer(coordsRawArray, dtype=np.float32,
                                   count=len(coordsRawArray)).reshape(self.nimgs, nma.pdb.n_atoms, 3)
        eta_global = frombuffer(etaRawArray, dtype=np.float64,
                                   count=len(etaRawArray)).reshape(self.nimgs, nma.nmodes_total)
        angles_global = frombuffer(anglesRawArray, dtype=np.float64,
                                   count=len(anglesRawArray)).reshape(self.nimgs, 3)
        shifts_global = frombuffer(shiftsRawArray, dtype=np.float64,
                                   count=len(shiftsRawArray)).reshape(self.nimgs, 3)
        nma_global = nma
        n_best_views_global = n_best_views
        plot_global = plot
        n_iter_global = n_iter
        gamma_global = gamma
        gamma_rigid_global = gamma_rigid

    def run_flexible_processes(self, workdata):
        rank = workdata[0]
        img = self.imgs[rank]

        if plot_global:
            plotter = FittingPlotter()
            plotter.start(img, img, [0.0], [0.0])
        else:
            plotter = None
        nmafits = []
        for v in range(n_best_views_global):
            tnma = nma_global.transform(self.rigid_angles[rank, v], self.rigid_shifts[rank, v])
            nmafit = NMAFit()
            nmafit.fit(img=img, nma=tnma, simulator=self.simulator, target_pdb=self.target_pdbs[rank],
                       n_iter=n_iter_global, gamma=gamma_global, gamma_rigid=gamma_rigid_global, verbose=False,
                       plot=plot_global, init_plotter=plotter)

            nmafits.append(nmafit)

        best_nma_idx = np.argmin([f.best_mse for f in nmafits])
        mses_global[rank] = nmafits[best_nma_idx].mse
        rmsds_global[rank] = nmafits[best_nma_idx].rmsd
        coords_global[rank] = nmafits[best_nma_idx].best_coords
        eta_global[rank] = nmafits[best_nma_idx].best_eta
        angles_global[rank] = nmafits[best_nma_idx].best_angle
        shifts_global[rank] = nmafits[best_nma_idx].best_shift

        if plot_global:
            plotter.update(nmafits[best_nma_idx].best_fitted_img, img, mses_global[rank], rmsds_global[rank],
                           mse_a=[f.mse for f in nmafits],
                           rmsd_a=[f.rmsd for f in nmafits])



