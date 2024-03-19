import unittest
from src.io import PDB
from src.utils import get_tests_data, get_tests_tmp, get_cc, get_angular_distance
from os.path import join
import numpy as np
from src.nma import NormalModesRTB
from src.simulator import AFMSimulator
from src.fitting import AFMFitting, rot_trans_match, ProjMatch, trans_match
from src.viewer import viewAFM
import polarTransform

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
        nma = NormalModesRTB.calculate_NMA(pdb, tmpDir=join(get_tests_tmp(), "nma"), nmodes=nmodes)

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

    def test_image_library(self):
        ref = PDB( join(get_tests_data(), "ref.pdb"))

        sim = AFMSimulator(size=40, vsize=7.0, beta=1.0, sigma=4.2, cutoff=20)
        library = sim.get_projection_library(pdb=ref,  angular_dist=20, verbose=False)
        self.assertEqual(78, library.nimg)

class TestFitting(unittest.TestCase):

    def test_nma_fitting(self):
        nma = NormalModesRTB.read_NMA(join(get_tests_data(), "nma"))
        ref = PDB( join(get_tests_data(), "ref.pdb"))
        target = PDB( join(get_tests_data(), "target.pdb"))

        zshift = 30.0
        sim = AFMSimulator(size=40, vsize=7.0, beta=1.0, sigma=4.2, cutoff=20)
        pexp = sim.pdb2afm(target, zshift=zshift)

        fit = AFMFitting(pdb=ref, img=pexp, simulator=sim, nma=nma, target_pdb=target)
        fit.fit_nma(n_iter=5, gamma=10, gamma_rigid=5,verbose=False, plot=False, zshift=30.0)
        self.assertGreater(fit.rmsd[0], 3.0)
        self.assertLess(fit.rmsd[-1], 1.0)

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

        shiftx, shifty, corr = trans_match(img1, img2)
        mse = np.sqrt(np.sum(np.square(img1)) + np.sum(np.square(img2)) - 2 * corr)
        rot2 = ref.copy()
        rot2.translate(np.array([shiftx, shifty, 0]) * sim.vsize)
        img3 = sim.pdb2afm(rot2, zshift=30.0)

        # assert that the MSE calculated by trans_match is the same than the MSE obtained by applying the shifts to
        #  the structure and projecting to image again
        self.assertAlmostEqual(np.linalg.norm(img3 - img2) , mse, 5)


    def test_rotational_matching(self):
        ref = PDB(join(get_tests_data(), "ref.pdb"))
        ref.center()
        angle_gt = np.array([200, 0, 0])
        shift_gt = np.array([-20, 30, 0])
        rot = ref.copy()
        rot.rotate(angle_gt)
        rot.translate(shift_gt)

        sim = AFMSimulator(size=40, vsize=7.0, beta=1.0, sigma=4.2, cutoff=20)
        pexp = sim.pdb2afm(rot)
        psim = sim.pdb2afm(ref)
        pexp2 = pexp.copy()
        pexp2[:7, :7] = 70

        angular_dist = 5
        angleSize = 2 ** 7

        angle, shift, mse, = rot_trans_match( psim,pexp,   angleSize)
        angle2, shift2, mse2 = rot_trans_match(psim,pexp2,  angleSize)

        self.assertLess(np.abs(angle -angle_gt[0])  , angular_dist )
        self.assertLess(np.abs(angle2 -angle_gt[0]) ,  angular_dist )
        self.assertLess(np.abs(shift[0] - shift_gt[0]/sim.vsize) , 1.0 )
        self.assertLess(np.abs(shift2[0] -shift_gt[0]/sim.vsize),  1.0 )
        self.assertLess(np.abs(shift[1] - shift_gt[1]/sim.vsize) , 1.0 )
        self.assertLess(np.abs(shift2[1] -shift_gt[1]/sim.vsize),  1.0 )

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
        library = sim.get_projection_library_full_pool(n_cpu=8, pdb=ref,  angular_dist=angular_dist, verbose=False, init_zshift=30.0,
                                             zshift_range=np.linspace(-15,15,10))
        pexp = sim.pdb2afm(rot, zshift_gt)


        projMatch = ProjMatch(img=pexp, simulator=sim, pdb=ref)
        projMatch.run(library=library, verbose=False)
        # projMatch.show()

        self.assertLess(np.abs(projMatch.best_shift[0] - shift_gt[0]) , sim.vsize)
        self.assertLess(np.abs(projMatch.best_shift[1] - shift_gt[1]) , sim.vsize)
        self.assertLess(np.abs(projMatch.best_shift[2] - zshift_gt) , 1.0)

        adist = get_angular_distance(projMatch.best_angle, angle_gt)
        self.assertLess(adist, angular_dist)

    def test_flexile_rigid_fitting(self):
        nma = NormalModesRTB.read_NMA(join(get_tests_data(), "nma"))
        ref = PDB( join(get_tests_data(), "ref.pdb"))
        target = PDB( join(get_tests_data(), "target.pdb"))
        angle_gt = np.array([200,-20,0])
        shift_gt = np.array([-15,10,0])
        zshift_gt = 41.6
        target.rotate(angle_gt)
        target.translate(shift_gt)

        sim = AFMSimulator(size=40, vsize=7.0, beta=1.0, sigma=4.2, cutoff=20)
        pexp = sim.pdb2afm(target, zshift=zshift_gt)
        pexp[:7, :7] = 70

        fit = AFMFitting(pdb=ref, img=pexp, simulator=sim, nma=nma, target_pdb=target)
        fit.fit_nma_rotations(n_iter=5, gamma=6, gamma_rigid=3,angular_dist=[20, 10, 5],
                near_angle_cutoff=[-1, 20, 10], n_views=[10, 5, 3], plot=False, verbose=False, zshift_range=[20, 10, 5],
                                                               zshift_points=[5, 5, 5])
        self.assertGreater(fit.rmsd[0], 3.0)
        self.assertLess(fit.best_rmsd, 2.1)
        self.assertLess(np.abs(fit.best_shift[0] - shift_gt[0]) , sim.vsize)
        self.assertLess(np.abs(fit.best_shift[1] - shift_gt[1]) , sim.vsize)
        self.assertLess(np.abs(fit.best_shift[2] - zshift_gt) , 1.0)
        self.assertLess(get_angular_distance(fit.best_angle, angle_gt), 10.0)


if __name__ == '__main__':
    unittest.main()
