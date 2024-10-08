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
from afmfit.pdbio import PDB
from afmfit.utils import matrix2euler, select_angles, get_flattest_angles
from afmfit.viewer import show_angular_distr, viewFit
from afmfit.utils import run_chimerax, to_mesh
from afmfit.simulator import AFMSimulator

import numpy as np
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool
import pickle
import tqdm
from multiprocessing.sharedctypes import RawArray
from numpy import frombuffer
import tempfile


class ProjMatch:
    def __init__(self, img, simulator, pdb):
        """
        Projection matching
        :param img: image to match size*size
        :param simulator: Simulator used for generating the projection library
        :param pdb: PDB used for generating the projection library
        """
        self.img = img
        self.simulator = simulator
        self.pdb = pdb
        self.size = simulator.size
        self.vsize = simulator.vsize

        self.angles = None
        self.shifts = None
        self.score = None
        self.best_angle = None
        self.best_shift = None
        self.best_score = None
        self.best_fitted_img = None

    def run(self, library, metric="rmsdp",  verbose=True, select_view_group=True):
        """
        Run projection matching
        :param library: ImageLibrary projection library
        :param metric: metric to use for the matching : mse | corr | cc | rmsdp
        :param verbose:
        :param select_view_group: if True, returns the min MSE angles and shifts for each view group
        """
        img_exp = self.img
        img_exp_norm = np.sum(np.square(img_exp))
        nimgs = library.nimg
        size = img_exp.shape[-1]

        # outputs
        scores =    np.zeros((nimgs))
        shifts = np.zeros((nimgs, 3))

        for i in tqdm.tqdm(range(nimgs), desc="Projection Matching", disable=not verbose):
            # Rotational Translational XY matching
            img = library.get_img(i)
            _x, _y, _c = self.trans_match(img,  img_exp)

            if metric == "mse":
                # calculate mse from correlation
                scores[i] = np.sum(np.square(img)) + img_exp_norm -2*_c
            elif metric == "corr":
                scores[i] = -_c
            elif metric == "cc":
                scores[i] = 1.0 -    _c / ( np.sqrt(np.sum(np.square(img))) * np.sqrt(img_exp_norm))
            elif metric == "rmsdp":
                scores[i] = np.sqrt(1/(size**2) * (np.sum(np.square(img)) + img_exp_norm -2*_c))


            # set the shifts
            shifts[i,0] =_x *  self.vsize
            shifts[i,1] =_y *  self.vsize
            shifts[i,2] =library.z_shifts[i]

        # select the best score for each view group
        if select_view_group:
            min_metric_group = library.view_group[
                (np.arange(library.nview), np.argmin(scores[library.view_group], axis=1))]
        else:
            min_metric_group = np.arange(nimgs)
        self.shifts = shifts[min_metric_group]
        self.scores = scores[min_metric_group]
        self.angles = library.angles[min_metric_group]

        # Outputs
        bestidx = np.argmin(self.scores)
        self.best_angle = self.angles[bestidx]
        self.best_shift = self.shifts[bestidx]
        fit = self.pdb.copy()
        fit.rotate(self.best_angle)
        fit.translate(self.best_shift)
        img = self.simulator.pdb2afm(fit, 0.0)
        self.best_score = self.scores[bestidx]
        self.best_fitted_img = img

        if verbose:
            print("Best score : %f"%self.best_score)

    def show(self, n_plot=3):
        """
        Show the n_plot first fitted images
        :param n_plot: number of images to show
        """
        fig, ax = plt.subplots(n_plot, 5, figsize=(20, 15))

        minscore = np.argsort(self.scores)
        for i in range(n_plot):
            fit = self.pdb.copy()
            fit.rotate(self.angles[minscore[i]])
            fit.translate(self.shifts[minscore[i]])
            pest = self.simulator.pdb2afm(fit, 0.0)
            score = np.linalg.norm(pest - self.img)
            ax[i, 0].imshow(pest.T, cmap="afmhot", origin="lower")
            ax[i, 1].imshow(self.img.T, cmap="afmhot", origin="lower")
            ax[i, 2].imshow(np.abs(self.img.T - pest.T), cmap="jet", origin="lower")
            ax[i, 3].plot(pest[self.size // 2], color="tab:red")
            ax[i, 3].plot(self.img[self.size // 2], color="tab:blue")
            ax[i, 4].plot(pest[:, self.size // 2], color="tab:red")
            ax[i, 4].plot(self.img[:, self.size // 2], color="tab:blue")
            ax[i, 0].set_title('Fitted [%.2f]' % (score))
            ax[i, 1].set_title('Input')
            ax[i, 2].set_title('Diff')
    def show_angular_distr(self):
        """
        Display a flatten projection of angular distribution and their correlation in the sphere
        """
        show_angular_distr(self.angles, color=self.scores, cmap = "jet", proj = "hammer", cbar = True)

    @classmethod
    def trans_match(cls, img1, img2):
        """
        Translational matching to find the best translation from img1 to img2
        :param img1:
        :param img2:
        :return: shiftx, shifty, correlatino
        """
        size = img1.shape[0]
        img1_ft = np.fft.rfftn(img1)
        img2_ft = np.fft.rfftn(img2)
        corr  = np.fft.irfftn(img1_ft * np.conjugate(img2_ft)).real
        dx, dy = np.unravel_index(corr.argmax(), corr.shape)
        max_corr = corr[dx,dy]
        if dx>= size//2:
            shiftx = size-dx
        else:
            shiftx = -dx
        if dy>= size//2:
            shifty = size-dy
        else:
            shifty = -dy
        return shiftx, shifty, max_corr

class NMAFit:
    def __init__(self):
        """
        Flexible Fitting of one image with NMA
        """
        self.best_score = None
        self.best_rmsd = None
        self.score = None
        self.rmsd = None
        self.best_eta = None
        self.best_fitted_img = None
        self.best_coords  = None
        self.best_angle = None
        self.best_shift = None

    def fit(self, img, nma, simulator, target_pdb=None, zshift=None,
            n_iter=10, lambda_r=100, lambda_f=100, verbose=True,q_init=None,):
        """
        Run the fitting
        :param img: Image to fit size*size
        :param nma: NormalModeRTB
        :param simulator: Simulator for pseudo-AFM images
        :param target_pdb: PDB to compute RMSD with, if None, the RMSD is compute with the initial model
        :param zshift: shift of the image in the z-axis
        :param n_iter: Number of iterations
        :param lambda_r: Lambda for rigid modes
        :param lambda_f: Lambda for flexible modes
        :param verbose: True to print progress bar
        :param q_init: Initial normal mode amplitude
        """
        dtimetot = time.time()

        # Inputs
        pdb = nma.pdb
        pexp = img
        size = pexp.shape[-1]
        if target_pdb is None:
            target_pdb = pdb
        if zshift is None:
            zshift=0.0

        # define output arrays
        dcd = np.zeros((n_iter+1, pdb.n_atoms, 3), dtype=np.float32)
        score = np.zeros(n_iter+1)
        rmsd =np.zeros(n_iter+1)
        eta = np.zeros((n_iter + 1, nma.nmodes_total))
        fitted_imgs =np.zeros((n_iter+1, simulator.size, simulator.size), dtype=np.float32)
        if q_init is not None:
            eta[0] = q_init

        # init state
        fitted = nma.applySpiralTransformation(eta=eta[0], pdb=pdb)
        psim = simulator.pdb2afm(fitted, zshift=zshift)

        score[0]= self.get_rmsdp(psim , pexp)

        rmsd[0]=fitted.getRMSD(target_pdb, align=True)
        dcd[0] = fitted.coords
        fitted_imgs[0] = psim

        # Define regularization
        regul = np.zeros( nma.nmodes_total)
        regul[:6] = 1 / (pdb.n_atoms * (lambda_r**2))
        regul[6:] = 1 / (pdb.n_atoms * (lambda_f**2))
        linear_modes_line =  nma.linear_modes.reshape(nma.nmodes_total, nma.natoms * 3)
        r_matrix = np.dot(linear_modes_line, linear_modes_line.T) * regul

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
            score[i+1] = self.get_rmsdp(psim , pexp)
            rmsd[i+1]= fitted.getRMSD(target_pdb, align=True)
            fitted_imgs[i+1]= psim

            # If the trajectory has converged, stop the iterations
            norm = np.array([np.linalg.norm(eta[i+1]-eta[q]) for q in range(i)])
            if any(norm<  1.0):
                for q in range(i, n_iter):
                    dcd[q + 1] = dcd[i+1]
                    score[q + 1] = score[i+1]
                    rmsd[q + 1] = rmsd[i+1]
                    fitted_imgs[q + 1] = fitted_imgs[i+1]
                break

            if verbose:
                print("Iter %i = %.2f ; %.2f" % (i, score[i+1], rmsd[i+1]))
                print(eta[i + 1])

        # outputs
        score = np.sqrt(score)
        best_idx= np.argmin(score)
        self.best_score = score[best_idx]
        self.best_rmsd= rmsd[best_idx]
        self.score = score
        self.rmsd= rmsd

        self.best_eta= eta[best_idx]
        self.best_fitted_img=fitted_imgs[best_idx]
        self.best_coords = dcd[best_idx]

        R, t = PDB.alignCoords(dcd[best_idx], pdb.coords)
        self.best_angle= matrix2euler(R.T)
        self.best_shift= t

        if verbose:
            print("\t TIME_TOTAL   => %.4f s" % (time.time() - dtimetot))

    def q_estimate(self, psim, pexp, dq, r_matrix, q0):
        """
        Estimate the NMA amplitude that fits psim into pexp
        :param psim: size*size
        :param pexp: size*size
        :param dq: Gradient of the image  nmodes* size*size
        :param r_matrix: regularization parameter (lambda) diagonal matrix
        :param q0: initial NMA amplitude
        :return: NMA amplitudes
        """
        nmodes = dq.shape[0]
        size = psim.shape[0]
        dy = (psim - pexp)
        dqt_line = dq.reshape(nmodes, size**2)
        dy_line = dy.reshape(size**2).T
        dqtdq = np.dot(dqt_line, dqt_line.T)
        dq2 = (1/size**2) * dqtdq + r_matrix
        dqtdy =  (1/size**2) * np.dot(dqt_line, dy_line)
        dqtdy += np.dot(r_matrix, q0)
        return - np.linalg.solve(dq2,dqtdy.T)

    def get_rmsdp(self, p1, p2):
        return (1 / p1.shape[-1] ** 2) * np.sum(np.square(p1 - p2))
class Fitter:
    def __init__(self, pdb, imgs, simulator, target_pdbs=None):
        """
        AFMFIT rigid and flexible fitting of AFM images
        :param pdb: PDB to fit
        :param imgs: set of AFM images to fit  nimgs*size*size
        :param simulator: Simulator
        :param target_pdbs: list of PDBs to compute RMSD with
        """
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
        self.rigid_scores = None
        self.rigid_imgs = None
        self.rigid_time = None
        self.rigid_done = False

        #Flexible outputs
        self.flexible_angles = None
        self.flexible_shifts = None
        self.flexible_eta = None
        self.flexible_scores = None
        self.flexible_rmsds = None
        self.flexible_coords = None
        self.flexible_imgs = None
        self.flexible_time = None
        self.flexible_done = False


    def dump(self, file):
        """
        Dump object to file
        :param file: pickle file
        """
        with open(file, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file):
        """
        Load object from pickled file
        :param file: pickle file
        :return: object
        """
        with open(file, "rb") as f:
            return pickle.load(f)

    def get_best_score(self):
        """
        Per image score
        :return:
        """
        return self.flexible_scores[(np.arange(self.nimgs), np.argmin(self.flexible_scores, axis=1))]
    def get_best_rmsd(self):
        """
        Per image RMSD
        :return:
        """
        return self.flexible_rmsds[(np.arange(self.nimgs), np.argmin(self.flexible_scores, axis=1))]

    def fit_rigid(self, n_cpu, angular_dist=10.0, verbose=False, zshift_range=None,
                  init_zshift=None, near_angle=None, near_angle_cutoff=None, select_view_group = True,
                  true_zshift=True, init_library=None, metric="rmsdp", init_angles=None):
        """
        Performs the rigid fitting on the set of images
        :param n_cpu: Number of CPUs
        :param angular_dist: Angular distance between projection images
        :param verbose:
        :param zshift_range: Range of shfits in the z-axis
        :param init_zshift: Z shift of the center of mass of the model (Ang)
        :param near_angle: three Euler angles. Restrict the projection matching to the neighborhood of specified angle
        :param near_angle_cutoff: Maximum angular distance (°) to restrict the projection matching around the specified angle
        :param select_view_group: if True, returns the min MSE angles and shifts for each view group
        :param true_zshift: if False, the z shift are estimated by shifting the entire image (faster but less accurate)
        :param init_library: uses an input library of images instead of generating a new one
        :param metric: metric to use for the fitting : mse | corr | cc | rmsdp
        :param init_angles: provide a list of projection angles
        """
        dt = time.time()

        #inputs
        if zshift_range is None:
            zshift_range = np.linspace(-20.0,20.0,10)

        # create library
        if init_library is None:
            library = self.simulator.project_library(n_cpu=n_cpu, pdb=self.pdb,  angular_dist=angular_dist, verbose=verbose,
                        zshift_range=zshift_range, near_angle=near_angle, near_angle_cutoff=near_angle_cutoff,
                                                     init_zshift=init_zshift, true_zshift=true_zshift, init_angles=init_angles)
        else:
            library = init_library

        if select_view_group:
            nview = library.nview
        else:
            nview = library.nimg

        workdata=[]
        for i in range(self.nimgs):
            workdata.append(
                [i]
            )
        # create output arrays
        anglesRawArray = RawArray("d", self.nimgs*3*nview)
        shiftsRawArray = RawArray("d", self.nimgs*3*nview)
        scoresRawArray = RawArray("d", self.nimgs*nview)
        imgsRawArray = RawArray("f", self.nimgs * self.simulator.size*self.simulator.size)

        # run
        p = Pool(n_cpu, initializer=self.init_rigid_processes, initargs=(library, anglesRawArray,
                    shiftsRawArray,scoresRawArray,imgsRawArray, self.nimgs, nview, self.simulator.size,
                    select_view_group, metric))
        for _ in tqdm.tqdm(p.imap_unordered(self.run_rigid_processes, workdata), total=len(workdata),
                           desc="Projection Matching", disable=not verbose):
            pass

        # free library
        if init_library is None:
            del library

        # set output arrays
        self.rigid_angles = frombuffer(anglesRawArray, dtype=np.float64,
                                   count=len(anglesRawArray)).reshape(self.nimgs,nview, 3)
        self.rigid_shifts = frombuffer(shiftsRawArray, dtype=np.float64,
                                   count=len(shiftsRawArray)).reshape(self.nimgs,nview, 3)
        self.rigid_scores = frombuffer(scoresRawArray, dtype=np.float64,
                                   count=len(scoresRawArray)).reshape(self.nimgs,nview)
        self.rigid_imgs = frombuffer(imgsRawArray, dtype=np.float32,
                                   count=len(imgsRawArray)).reshape(self.nimgs,self.simulator.size,self.simulator.size)
        self.rigid_time = time.time() - dt
        self.rigid_done = True


    def fit_flexible(self, n_cpu, nma, verbose=False, n_best_views=20,dist_views=20,
                     n_iter=10, lambda_f=10, lambda_r=None):
        """
        Performs the flexible fitting on the set of images
        :param n_cpu: Number of CPUs
        :param nma: NormalModeRTB
        :param verbose: print progress bar
        :param n_best_views: Number of views from the rigid fitting to process
        :param dist_views: Angular distance between views (°)
        :param n_iter: number of iteration of the NMA flexible fitting
        :param lambda_f: Lambda for flexible modes
        :param lambda_r: Lambda for rigid modes if specified, else lambda_f
        """
        dt = time.time()
        if not self.rigid_done:
            raise RuntimeError("Perform rigid fitting first")
        if lambda_r is None:
            lambda_r = lambda_f

        # Arrays
        scoresRawArray   = RawArray("d", self.nimgs * (n_iter+1))
        rmsdsRawArray  = RawArray("d", self.nimgs * (n_iter+1))
        coordsRawArray = RawArray("f", self.nimgs * nma.pdb.n_atoms * 3)
        etaRawArray    = RawArray("d", self.nimgs * nma.nmodes_total)
        anglesRawArray = RawArray("d", self.nimgs*3)
        shiftsRawArray = RawArray("d", self.nimgs*3)
        imgsRawArray   = RawArray("f", self.nimgs * self.simulator.size*self.simulator.size)

        workdata = []
        for i in range(self.nimgs):
            workdata.append([i])

        if n_best_views is None:
            n_best_views = self.rigid_angles.shape[1]

        # Create pool
        if n_cpu >1:
            p = Pool(n_cpu, initializer=self.init_flexible_processes, initargs=(rmsdsRawArray,anglesRawArray,
                                    shiftsRawArray, scoresRawArray, coordsRawArray, etaRawArray, imgsRawArray,
                                    nma, n_iter, n_best_views, dist_views, lambda_r, lambda_f))

            # Run Pool
            for _ in tqdm.tqdm(p.imap_unordered(self.run_flexible_processes, workdata), total=len(workdata), desc="Flexible Fitting"
                               , disable=not verbose):
                pass
        else:
            self.init_flexible_processes(rmsdsRawArray, anglesRawArray,
                                                      shiftsRawArray, scoresRawArray, coordsRawArray, etaRawArray,
                                                      imgsRawArray,
                                                      nma, n_iter, n_best_views, dist_views, lambda_r, lambda_f)
            for w in workdata:
                self.run_flexible_processes(w)


        # Outputs
        self.flexible_scores = frombuffer(scoresRawArray, dtype=np.float64,
                                   count=len(scoresRawArray)).reshape(self.nimgs, (n_iter+1))
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
        self.flexible_imgs = frombuffer(imgsRawArray, dtype=np.float32,
                                   count=len(imgsRawArray)).reshape(self.nimgs,self.simulator.size,self.simulator.size)

        # Fix the angles and shifts according to displacements
        for i in range(self.nimgs):
            R, t = PDB.alignCoords( self.flexible_coords[i], self.pdb.coords)
            self.flexible_angles[i] = matrix2euler(R.T)
            self.flexible_shifts[i] = t
        self.flexible_time = time.time() - dt
        self.flexible_done=True


    def init_rigid_processes(self, library,anglesRawArray,shiftsRawArray, scoresRawArray,imgsRawArray, nimgs,
                                 nview,size, select_view_group, metric):
        global library_global
        global angles_global
        global shifts_global
        global scores_global
        global imgs_global
        global select_view_group_global
        global metric_global
        library_global = library

        angles_global = frombuffer(anglesRawArray, dtype=np.float64,
                                   count=len(anglesRawArray)).reshape(nimgs,nview,3)

        shifts_global = frombuffer(shiftsRawArray, dtype=np.float64,
                                   count=len(shiftsRawArray)).reshape(nimgs,nview, 3)
        scores_global = frombuffer(scoresRawArray, dtype=np.float64,
                                   count=len(scoresRawArray)).reshape(nimgs,nview)
        imgs_global = frombuffer(imgsRawArray, dtype=np.float32,
                                   count=len(imgsRawArray)).reshape(nimgs, size, size)
        select_view_group_global = select_view_group
        metric_global = metric


    def run_rigid_processes(self, workdata):
        """
        Runs the rigid fitting in parallel
        :param workdata:
        """
        rank = workdata[0]

        projMatch = ProjMatch(img=self.imgs[rank], simulator=self.simulator, pdb=self.pdb)
        projMatch.run(library=library_global, verbose=False, select_view_group=select_view_group_global,
                      metric=metric_global)

        idx_best_views =np.argsort(projMatch.scores)
        angles_global[rank,:] = projMatch.angles[idx_best_views]
        shifts_global[rank,:] = projMatch.shifts[idx_best_views]
        scores_global[rank,:] = projMatch.scores[idx_best_views]
        imgs_global[rank] = projMatch.best_fitted_img

    def init_flexible_processes(self, rmsdsRawArray,anglesRawArray,shiftsRawArray, scoresRawArray, coordsRawArray, etaRawArray,
                                imgsRawArray, nma, n_iter, n_best_views, dist_views, lambda_r, lambda_f):
        global imgs_global
        global eta_global
        global rmsds_global
        global coords_global
        global angles_global
        global shifts_global
        global scores_global
        global n_best_views_global
        global dist_views_global
        global nma_global
        global n_iter_global
        global lambda_r_global
        global lambda_f_global
        scores_global = frombuffer(scoresRawArray, dtype=np.float64,
                                   count=len(scoresRawArray)).reshape(self.nimgs, (n_iter+1))
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
        imgs_global = frombuffer(imgsRawArray, dtype=np.float32,
                                   count=len(imgsRawArray)).reshape(self.nimgs, self.simulator.size, self.simulator.size)
        nma_global = nma
        n_best_views_global = n_best_views
        dist_views_global = dist_views
        n_iter_global = n_iter
        lambda_r_global = lambda_r
        lambda_f_global = lambda_f

    def run_flexible_processes(self, workdata):
        """
        Runs the flexible fitting in parallel
        :param workdata:
        """
        rank = workdata[0]

        nmafits = []

        # Select the views to process
        if dist_views_global is None:
            views_idx = np.arange(n_best_views_global)
        else:
            views_idx = select_angles(self.rigid_angles[rank], n_points=n_best_views_global, threshold=dist_views_global)

        # Loop over each view
        for v in views_idx:
            # Rotate the modes
            tnma = nma_global.transform(self.rigid_angles[rank, v], self.rigid_shifts[rank, v])

            # Fit the image with NMA
            nmafit = NMAFit()
            nmafit.fit(img=self.imgs[rank], nma=tnma, simulator=self.simulator, target_pdb=self.target_pdbs[rank],
                       n_iter=n_iter_global, lambda_r=lambda_r_global, lambda_f=lambda_f_global, verbose=False)
            nmafits.append(nmafit)

        # Outputs
        best_nma_idx = np.argmin([f.best_score for f in nmafits])
        scores_global[rank] = nmafits[best_nma_idx].score
        rmsds_global[rank] = nmafits[best_nma_idx].rmsd
        coords_global[rank] = nmafits[best_nma_idx].best_coords
        eta_global[rank] = nmafits[best_nma_idx].best_eta
        angles_global[rank] = nmafits[best_nma_idx].best_angle
        shifts_global[rank] = nmafits[best_nma_idx].best_shift
        imgs_global[rank] = nmafits[best_nma_idx].best_fitted_img

    def show_rigid(self):
        """
        Show the angular distribution of the rigid fitting
        """
        if not self.rigid_done:
            raise RuntimeError("Perform fitting first")
        show_angular_distr(self.rigid_angles[:,0], np.sqrt(self.rigid_scores[:,0]), cbar=True)
    def show(self, **kwargs):
        """
        Opens the fitting viewer
        """
        viewFit(self, **kwargs)

    def viewFitChimera(self, index=0, translate_z=0.0, upsample=2, truemesh=True):
        """
        Shows the one fit in ChimeraX
        :param index: Index of the image to show
        :param translate_z: Translation in the z-axis
        :param upsample: Upsample the image
        :param truemesh: Shows the image as a mesh or clouds of points
        """
        if not self.flexible_done:
            raise RuntimeError("Perform fitting first")
        with tempfile.TemporaryDirectory() as tmpdir:
            to_mesh(img=self.imgs[index], vsize=self.simulator.vsize, file=tmpdir + "/img.bild", upsample=upsample,
                    truemesh=truemesh)
            pdb = self.pdb.copy()
            pdb.rotate(self.rigid_angles[index, 0])
            pdb.translate(self.rigid_shifts[index, 0])
            pdb.translate([0.0, 0.0, -translate_z])
            pdb.write_pdb(tmpdir + "/rigid.pdb")
            pdb.coords = self.flexible_coords[index]
            pdb.translate([0.0, 0.0, -translate_z])
            pdb.write_pdb(tmpdir + "/flex.pdb")
            with open(tmpdir + "/cmd.cxc", "w") as f:
                f.write("open %s/rigid.pdb \n" % tmpdir)
                f.write("open %s/flex.pdb \n" % tmpdir)
                f.write("open %s/img.bild \n" % tmpdir)
                f.write("morph #1-2 same t\n")
                f.write("set bgColor white \n")
            run_chimerax(tmpdir + "/cmd.cxc")

    def get_stats(self):
        """
        Get stats from the flexible fitting
        :return: score_rigid, score_flexible, rmsd, % of decrease
        """
        if self.flexible_scores is None:
            raise RuntimeError("Perform fitting first")
        scorer  = self.flexible_scores[:,0]
        scoref  = np.min(self.flexible_scores, axis=1)
        rmsd  = self.flexible_rmsds[(np.arange(self.nimgs),np.argmin(self.flexible_scores, axis=1))]
        def get_pct(vr, vf):
            return 100*(1- (vr/vf))
        return scorer.mean(), (scoref ).mean(), rmsd.mean(), get_pct((scorer).mean(),(scoref ).mean())

    def show_stats(self):
        """
        Prints stats from the flexible fitting
        """
        scorer, scoref, rmsd, pct = self.get_stats()
        print("Best score rigid : %.2f "%(scorer))
        print("Best score flexible : %.2f "%(scoref))
        print("score decrease : %.2f %%"%(pct))
        print("Best RMSD: %.2f Ang "%(rmsd))



def sigma_estimate(imgs, pdb, n_cpu=4, angular_dist=10, metric="rmsdp",quality_ratio=8.0,
                    min_sigma=2.0, max_sigma=10.0, resolution=0.5, angle_percent=20.0, zshift_range=None,
                    init_zshift=None, near_angle=None, near_angle_cutoff=None, true_zshift=False):
    if near_angle is None:
        angle = get_flattest_angles(pdb, percent=angle_percent, angular_dist=angular_dist)
    else:
        angle=None

    sigma_linspace = np.arange(min_sigma,max_sigma, resolution)

    val = []
    for i in tqdm.tqdm(range(len(sigma_linspace))):
        sigma = sigma_linspace[i]
        sim = AFMSimulator(size=imgs.sizex, vsize=imgs.vsize, sigma=sigma,
                           quality_ratio=quality_ratio)
        proj = sim.project_library(
            n_cpu=n_cpu, pdb=pdb, angular_dist=angular_dist, init_zshift=init_zshift, near_angle=near_angle, near_angle_cutoff=near_angle_cutoff,
            verbose=True, true_zshift=true_zshift,
            zshift_range=zshift_range, init_angles=angle)
        fitter = Fitter(pdb=pdb, simulator=sim, imgs=imgs.imgs)
        fitter.fit_rigid(n_cpu=n_cpu, angular_dist=angular_dist, verbose=True, zshift_range=zshift_range,
                         init_zshift=init_zshift, near_angle=near_angle, near_angle_cutoff=near_angle_cutoff, select_view_group=True,
                         true_zshift=true_zshift, init_library=proj, metric=metric)
        val.append(np.mean(fitter.rigid_scores[:,0]))
        del proj
        del fitter

    plt.figure()
    plt.plot(sigma_linspace, val)
    plt.show()

    return sigma_linspace[np.argmin(val)], sigma_linspace, val