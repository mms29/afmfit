import unittest
from src.io import PDB
from src.utils import get_tests_data, get_tests_tmp, get_cc, get_angular_distance
from os.path import join
import numpy as np
from src.nma import NormalModesRTB
from src.simulator import AFMSimulator
from src.fitting import AFMFitting

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
        psim = sim.pdb2afm(ref, zshift=30.0)
        pexp = sim.pdb2afm(target, zshift=30.0)

        self.assertAlmostEqual(0.99, get_cc(psim, pexp), places=3)

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
        dcd, mse, rmsd, qt = fit.fit_nma(n_iter=5, gamma=10, gamma_rigid=5,verbose=False, plot=False, zshift=30.0)
        self.assertTrue(rmsd[0]> 3.0)
        self.assertTrue(rmsd[-1]< 1.0)

    def test_proj_match(self):
        angular_dist = 20

        ref = PDB( join(get_tests_data(), "ref.pdb"))
        # target = PDB( join(get_tests_data(), "target.pdb"))

        sim = AFMSimulator(size=40, vsize=7.0, beta=1.0, sigma=4.2, cutoff=20)
        library = sim.get_projection_library(pdb=ref,  angular_dist=angular_dist, verbose=False)
        pexp = sim.pdb2afm(ref)

        fit = AFMFitting(pdb=ref, img=pexp, simulator=sim)
        angles, shifts, mse = fit.projection_matching(image_library=library, angular_dist=angular_dist, max_shift_search=5,
                                verbose=False, plot = False)
        min_angle = [get_angular_distance(angles[np.argsort(mse)][i], np.zeros(3)) for i in range(len(mse))]
        self.assertEqual(0, np.argmin(min_angle))

    def test_flexile_rigid_fitting(self):
        nma = NormalModesRTB.read_NMA(join(get_tests_data(), "nma"))
        ref = PDB( join(get_tests_data(), "ref.pdb"))
        target = PDB( join(get_tests_data(), "target.pdb"))

        zshift = 30.0
        sim = AFMSimulator(size=40, vsize=7.0, beta=1.0, sigma=4.2, cutoff=20)
        pexp = sim.pdb2afm(target, zshift=zshift)

        fit = AFMFitting(pdb=ref, img=pexp, simulator=sim, nma=nma, target_pdb=target)
        dcds, mses, rmsd, q_est, psim = fit.fit_nma_rotations(solver_iter=5, gamma=10, gamma_rigid=5,
                                                               max_shift_search=5, angular_dist=[20, 10, 5],
                                                               near_cutoff=[-1, 20, 10], n_points=[10, 5, 3],
                                                               plot=False, verbose=False, zshift_range=[10, 5, 2],
                                                               zshift_points=[5, 5, 5])
        self.assertTrue(rmsd[0]> 3.0)
        self.assertTrue(rmsd[-1]< 1.0)


if __name__ == '__main__':
    unittest.main()
