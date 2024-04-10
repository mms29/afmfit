import unittest
from afmfit.pdbio import PDB
from afmfit.utils import get_tests_data, get_tests_tmp, get_cc, get_angular_distance
from os.path import join
import numpy as np
from afmfit.nma import NormalModesRTB
from afmfit.simulator import AFMSimulator
from afmfit.fitting import Fitter, ProjMatch, NMAFit
from afmfit.viewer import viewAFM
import polarTransform
import multiprocessing

N_CPU_TOTAL = multiprocessing.cpu_count()//2

class TestIO(unittest.TestCase):

    def test_io_pdb(self):
        pdb_file = join(get_tests_data(), "ref.pdb")
        pdb_file2 = join(get_tests_tmp(), "tmp.pdb")
        pdb = PDB(pdb_file)
        pdb.write_pdb(pdb_file2)
        pdb2 = PDB(pdb_file2)

        self.assertEqual(np.sum(pdb.coords - pdb2.coords), 0.0)
        self.assertTrue(all(pdb.resNum == pdb2.resNum))
        self.assertTrue(all(pdb.resName == pdb2.resName))
        self.assertTrue(all(pdb.atomName == pdb2.atomName))
        self.assertTrue(all(pdb.atomNum == pdb2.atomNum))

class TestNMA(unittest.TestCase):

    def test_calculate_nma(self):
        nmodes = 10
        pdb = PDB( join(get_tests_data(), "ref.pdb"))
        nma = NormalModesRTB.calculate_NMA(pdb, prefix=join(get_tests_tmp(), "nma"), nmodes=nmodes)

        eta = np.zeros(nma.nmodes_total)
        eta[6] = 2000

        #Spiral transform
        new_pdb = nma.applySpiralTransformation(eta, pdb)
        self.assertAlmostEqual(6.92, pdb.getRMSD(new_pdb), places=2)

        #Linear Transform
        new_pdb_lin = nma.applyLinearTransformation(eta, pdb)
        self.assertAlmostEqual(6.99, pdb.getRMSD(new_pdb_lin), places=2)

    def test_read_nma(self):
        nma = NormalModesRTB.read_NMA(join(get_tests_data(), "nma"))
        pdb = PDB( join(get_tests_data(), "nma.pdb"))

        eta = np.zeros(nma.nmodes_total)
        eta[6] = 2000

        #Spiral transform
        new_pdb = nma.applySpiralTransformation(eta, pdb)
        self.assertAlmostEqual(6.92, pdb.getRMSD(new_pdb), places=2)

        #Linear Transform
        new_pdb_lin = nma.applyLinearTransformation(eta, pdb)
        self.assertAlmostEqual(6.99, pdb.getRMSD(new_pdb_lin), places=2)


class TestSimulator(unittest.TestCase):

    def test_simulator(self):
        ref = PDB( join(get_tests_data(), "ref.pdb"))
        target = PDB( join(get_tests_data(), "target.pdb"))

        sim = AFMSimulator(size=40, vsize=7.0, beta=1.0, sigma=4.2, cutoff=20)
        img1 = sim.pdb2afm(ref, zshift=30.0)
        img2 = sim.pdb2afm(target, zshift=30.0)

        img1_gt = np.loadtxt(join(get_tests_data(), "ref_img.tsv")).T
        img2_gt = np.loadtxt(join(get_tests_data(), "target_img.tsv")).T

        self.assertLess(get_cc(img2, img1_gt), get_cc(img1, img1_gt))
        self.assertLess(get_cc(img1, img2_gt), get_cc(img2, img2_gt))


class TestFitting(unittest.TestCase):

    def test_nma_fitting(self):
        nma = NormalModesRTB.read_NMA(join(get_tests_data(), "nma"))
        target = PDB( join(get_tests_data(), "target.pdb"))

        zshift = 30.0
        sim = AFMSimulator(size=40, vsize=7.0, beta=1.0, sigma=4.2, cutoff=20)
        pexp = sim.pdb2afm(target, zshift=zshift)

        nmafit = NMAFit()
        nmafit.fit(img=pexp,nma=nma, simulator=sim,target_pdb=target, zshift=zshift,
                   n_iter=5, gamma=10, gamma_rigid=5,verbose=False, plot=False)
        self.assertGreater(nmafit.rmsd[0], 3.0)
        self.assertLess(nmafit.rmsd[-1], 1.0)

    def test_trans_match(self):
        ref = PDB(join(get_tests_data(), "ref.pdb"))
        ref.center()
        shift_gt = [28.5, -33.72, 0]
        rot = ref.copy()
        rot.translate(shift_gt)

        sim = AFMSimulator(size=40, vsize=7.23, beta=1.0, sigma=4.2, cutoff=20)
        img1 = sim.pdb2afm(ref, zshift=30.0)
        img2 = sim.pdb2afm(rot, zshift=30.0)
        viewAFM([img1, img2])

        shiftx, shifty, corr = ProjMatch.trans_match(img1, img2)
        mse = np.sqrt(np.sum(np.square(img1)) + np.sum(np.square(img2)) - 2 * corr)
        rot2 = ref.copy()
        rot2.translate(np.array([shiftx, shifty, 0]) * sim.vsize)
        img3 = sim.pdb2afm(rot2, zshift=30.0)

        # assert that the MSE calculated by trans_match is the same than the MSE obtained by applying the shifts to
        #  the structure and projecting to image again
        self.assertAlmostEqual(np.linalg.norm(img3 - img2) , mse, 2)

    def test_proj_match(self):
        angular_dist = 10

        ref = PDB( join(get_tests_data(), "ref.pdb"))
        ref.center()
        angle_gt = [200,60,-60]
        shift_gt = [10,-5,0]
        zshift_gt = 41.6
        rot = ref.copy()
        rot.rotate(angle_gt)
        rot.translate(shift_gt)

        sim = AFMSimulator(size=40, vsize=7.0, beta=1.0, sigma=4.2, cutoff=20)
        library = sim.project_library(n_cpu=N_CPU_TOTAL, pdb=ref,  angular_dist=angular_dist, verbose=True, init_zshift=30.0,
                                             zshift_range=np.linspace(-15,15,10))
        pexp = sim.pdb2afm(rot, zshift_gt)


        projMatch = ProjMatch(img=pexp, simulator=sim, pdb=ref)
        projMatch.run(library=library, verbose=True)
        # projMatch.show()

        self.assertLess(np.abs(projMatch.best_shift[0] - shift_gt[0]) , sim.vsize)
        self.assertLess(np.abs(projMatch.best_shift[1] - shift_gt[1]) , sim.vsize)
        self.assertLess(np.abs(projMatch.best_shift[2] - zshift_gt) , 1.0)

        adist = get_angular_distance(projMatch.best_angle, angle_gt)
        self.assertLess(adist, angular_dist)


    def test_fitting_rigid(self):
        angular_dist = 15

        ref = PDB( join(get_tests_data(), "ref.pdb"))
        ref.center()
        nimg = 10
        angle_gt = np.zeros((nimg,3))
        shift_gt = np.zeros((nimg,3))
        angle_gt[:,0] = np.linspace(0,360, nimg)
        angle_gt[:,1] = np.linspace(-20,20, nimg)
        shift_gt[:,0] = np.linspace(-20,20, nimg)
        shift_gt[:,1] = np.linspace(-20,20, nimg)
        shift_gt[:,2] = 41.6
        sim = AFMSimulator(size=40, vsize=7.0, beta=1.0, sigma=4.2, cutoff=20)
        zshift_range = np.linspace(-20,20,10)

        imgs = []
        targets = []
        for i in range( nimg):
            rot = ref.copy()
            rot.rotate(angle_gt[i])
            rot.translate(shift_gt[i])
            targets.append(rot)
            imgs.append(sim.pdb2afm(rot, 0.0))

        fitter = Fitter(pdb=ref, imgs=imgs, simulator=sim, target_pdbs=targets)
        fitter.fit_rigid( n_cpu=N_CPU_TOTAL, angular_dist=angular_dist, verbose=True, zshift_range=zshift_range)

        for i in range(nimg):
            adist = get_angular_distance(fitter.rigid_angles[i,0], angle_gt[i])
            sdist = np.linalg.norm(fitter.rigid_shifts[i,0]-shift_gt[i])
            print(adist)
            print(sdist)
            self.assertLess(adist, 21.0)
            self.assertLess(sdist, 6.0)

    def test_fitting_flexible(self):
        nma = NormalModesRTB.read_NMA(join(get_tests_data(), "nma"))
        nimg = 10
        sim = AFMSimulator(size=40, vsize=7.0, beta=1.0, sigma=3.2, cutoff=40)
        zshift_range = np.linspace(-20,20,10)

        angle_gt = np.zeros((nimg,3))
        shift_gt = np.zeros((nimg,3))
        eta_gt = np.zeros((nimg, nma.nmodes_total))
        angle_gt[:,0] = np.linspace(0,360, nimg)
        angle_gt[:,1] = np.linspace(-20,20, nimg)
        shift_gt[:,0] = np.linspace(-20,20, nimg)
        shift_gt[:,1] = np.linspace(-20,20, nimg)
        shift_gt[:,2] = 41.6
        eta_gt[:, 6] = np.linspace(-1000,-500, nimg)
        eta_gt[:, 7] = np.linspace(500,1000, nimg)

        imgs = []
        targets = []
        for i in range( nimg):
            tnma = nma.transform(angle_gt[i], shift_gt[i])
            target = tnma.applySpiralTransformation(eta_gt[i], tnma.pdb)
            targets.append(target)
            imgs.append(sim.pdb2afm(target, 0.0))

        # viewAFM(imgs, interactive=True)

        fitter = Fitter(pdb=nma.pdb, imgs=imgs, simulator=sim, target_pdbs=targets)
        fitter.fit_rigid( n_cpu=N_CPU_TOTAL, angular_dist=10, verbose=True, zshift_range=zshift_range)
        fitter.fit_flexible( n_cpu=N_CPU_TOTAL, nma=nma, verbose=True, n_best_views=3,
                     n_iter=10, gamma=10, gamma_rigid=3, plot=False)

        for i in range(nimg):
            self.assertLess(3.0, fitter.flexible_rmsds[i,0])
            self.assertLess(fitter.flexible_rmsds[i].min(), 2.5)

            adist = get_angular_distance(fitter.flexible_angles[i], angle_gt[i])
            sdist = np.linalg.norm(fitter.flexible_shifts[i]-shift_gt[i])
            self.assertLess(adist, 11.0)
            self.assertLess(sdist, 3.0)

if __name__ == '__main__':
    unittest.main()
