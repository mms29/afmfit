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
from afmfit.pdbio import  numpyArr2dcd
from afmfit.utils import euler2matrix

import numpy as np
import os
from os.path import join
from numba import njit
from afmfit.utils import get_nolb_path, run_chimerax
import tempfile

class NormalModesRTB:
    def __init__(self,pdb, linear_modes, m_rigidT, m_rigidR, mapping, mapping_len, com):
        """
        Normal modes RTB using NOLB
        :param pdb: Initial model
        :param linear_modes: Linear NMA
        :param m_rigidT: rigid translations of the RTBs
        :param m_rigidR: rigid rotations of the RTBs
        :param mapping: Mapping RTB to all atom
        :param mapping_len: Mapping length
        :param com: Centers of mass of the RTB
        """
        self.pdb = pdb
        self.linear_modes = linear_modes
        self.m_rigidT = m_rigidT
        self.m_rigidR = m_rigidR
        self.mapping = mapping
        self.mapping_len = mapping_len
        self.com = com
        self.nmodes_total = self.linear_modes.shape[0]
        self.natoms = self.linear_modes.shape[1]
        self.nmodes = self.nmodes_total -6
        self.nrtb = len(com)

    def select_modes(self, idx:int):
        """
        Select modes
        :param idx: list of indexes
        """
        idx = np.array(idx)
        self.linear_modes = self.linear_modes[idx]
        self.m_rigidT = self.m_rigidT[:,idx]
        self.m_rigidR = self.m_rigidR[:,idx]
        self.nmodes_total = self.linear_modes.shape[0]
        self.nmodes = self.nmodes_total -6


    @classmethod
    def read_NMA(cls, prefix:str):
        """
        Read NMA files from NOLB from the "--format 3" option
        :param prefix: prefix of the NOLB files
        :return: NormalModesRTB
        """
        pdb = PDB(prefix+".pdb")
        m_rigidR, m_rigidT, com = cls._read_rtb(prefix + "_rtb.txt")
        mapping, mapping_len = cls._read_mapping(prefix + "_rtb-mapping.txt")
        linear_modes = cls._read_linear_modes(prefix + "_linear_modes.txt")
        return cls(pdb, linear_modes, m_rigidT, m_rigidR, mapping, mapping_len, com)

    @classmethod
    def calculate_NMA(cls, pdb, nmodes, prefix = None, cutoff = 8.0, options = ""):
        """
        Calculate NMA with NOLB
        :param pdb: Atomic model PDB
        :param nmodes: number of modes
        :param prefix: output prefix
        :param cutoff: cutoff distance (Ang)
        :param options:  other options to pass to NOLB
        :return: NormalModesRTB
        """
        with tempfile.TemporaryDirectory() as tmpDir :
            tmpPrefix = join(tmpDir, "")
            if prefix is not None:
                tmpPrefix=prefix
            if isinstance(pdb, PDB):
                pdbfile = tmpPrefix+".pdb"

                pdb.write_pdb(pdbfile)
            else:
                pdbfile = pdb
                try:
                    pdb = PDB(pdbfile)
                except:
                    pass

            # Run NOLB
            cmd = "%s %s -o %s -s 0 -n %i --format 3 -c %f %s" %(get_nolb_path(), pdbfile, tmpPrefix, nmodes, cutoff, options)
            if os.system(cmd):
                raise RuntimeError("NOLB execution failed")

            # Read outputs
            m_rigidR, m_rigidT, com = cls._read_rtb( tmpPrefix+ "_rtb.txt")
            mapping, mapping_len = cls._read_mapping(tmpPrefix+ "_rtb-mapping.txt")
            linear_modes = cls._read_linear_modes(   tmpPrefix+ "_linear_modes.txt")

        return cls(pdb, linear_modes, m_rigidT, m_rigidR, mapping, mapping_len, com)

    def viewChimera(self, amp=1000, npoints= 10):
        """
        Show modes in ChimeraX
        :param amp: Amplitude
        :param npoints: number of trajectory points
        """
        dcd = np.zeros((self.nmodes, npoints, self.natoms,3))

        with tempfile.TemporaryDirectory() as tmpDir:
            pdbfile = join( tmpDir,"ref.pdb")
            self.pdb.write_pdb(pdbfile)
            for m in range(self.nmodes):
                q_range = np.linspace(-amp, amp, npoints)
                for i in range(npoints):
                    q = np.zeros(self.nmodes_total)
                    q[m + 6] = q_range[i]
                    transformed = self.applySpiralTransformation(q, self.pdb)
                    dcd[m, i] = transformed.coords

                numpyArr2dcd(arr=dcd[m], filename=join(tmpDir,"mode%i.dcd"%(m+7)))
            cmd_file = join(tmpDir,"cmd.cxc")
            with open(cmd_file ,"w") as f:
                for m in range(self.nmodes):
                    f.write("open %s name %s \n" %(pdbfile, "mode%i"%(m+7)))
                    f.write("open %s structureModel #%i \n" %(join(tmpDir,"mode%i.dcd"%(m+7)), m+1))
                f.write("hide all models\n")
                f.write("show #1 models\n")
            run_chimerax(cmd_file)

    def transform(self, angle, shift):
        """
        Rigid transformation of the modes
        :param angle: three Euler angles for 3D rotation
        :param shift: three translations
        :return: Transformed NormalModesRTB
        """
        rot_m_rigidR = np.zeros(self.m_rigidR.shape)
        rot_m_rigidT = np.zeros(self.m_rigidT.shape)
        rot_linear_modes = np.zeros(self.linear_modes.shape)

        # Get rotation matrix
        R = euler2matrix(angle)
        Rt = R.T

        # Rotate and shift center of mass
        rot_com = np.dot(self.com, Rt) + shift

        #Rotate and shift PDB
        rot_pdb = self.pdb.copy()
        rot_pdb.rotate(angle)
        rot_pdb.translate(shift)

        # Rotate only the normal modes
        for m in range(self.nmodes_total):
            rot_m_rigidR[:, m] = np.dot(self.m_rigidR[:, m], Rt)
            rot_m_rigidT[:, m] = np.dot(self.m_rigidT[:, m], Rt)
            rot_linear_modes[m] = np.dot(self.linear_modes[m], Rt)

        return NormalModesRTB(rot_pdb, rot_linear_modes, rot_m_rigidT, rot_m_rigidR, self.mapping, self.mapping_len, rot_com)

    @classmethod
    def _read_rtb(cls, rtb_file):
        rtb = np.loadtxt(rtb_file)
        nrigid = rtb.shape[0]
        nmodes = (rtb.shape[1] - 4) // 6
        caId = rtb[:, 0]
        com = rtb[:, 1:4]
        m_rigidR = np.zeros((nrigid, nmodes, 3))
        m_rigidT = np.zeros((nrigid, nmodes, 3))
        for i in range(nrigid):
            for m in range(nmodes):
                idstart = 4 + m * 6
                m_rigidR[i, m] = rtb[i, idstart:(idstart + 3)]
                m_rigidT[i, m] = rtb[i, (idstart + 3):(idstart + 6)]
        return m_rigidR, m_rigidT, com

    @classmethod
    def _read_linear_modes(cls, linear_modes_file):
        with open(linear_modes_file, "r") as f:
            linear_modes = []
            mode = []
            for line in f:
                spl = line.split()
                if len(spl) != 3:
                    if len(mode) != 0:
                        linear_modes.append(mode)
                        mode = []
                else:
                    mode.append([float(spl[0]), float(spl[1]), float(spl[2])])
            linear_modes.append(mode)
        linear_modes_arr = np.array(linear_modes)
        return linear_modes_arr

    @classmethod
    def _read_mapping(cls, mapping_file):
        mapping_len = []
        with open(mapping_file, "r") as f:
            mapping = []
            for line in f:
                res = np.array(line.split(), dtype=int)[2:]
                mapping_len.append(len(res))
                mapping.append(res)
        mapping_len = np.array(mapping_len)
        maxlen = mapping_len.max()
        nres = mapping_len.shape[0]
        mapping_arr = np.zeros((nres, maxlen), dtype=int)
        for caId in range(nres):
            mapping_arr[caId, :mapping_len[caId]] = mapping[caId]

        return mapping_arr, mapping_len

    def applySpiralTransformation(self, eta, pdb=None):
        """
        Apply spiral (nonlinear) transformation
        :param eta: array of n_modes_total with the amplitudes
        :param pdb: PDB on which the transformation is applied
        :return: PDB
        """
        if pdb is None:
            pdb=self.pdb
        transformed_pdb = pdb.copy()
        new_coords = applySpiralTransformation_njit(eta, transformed_pdb.coords,
            nmodes= self.nmodes, nmodes_total= self.nmodes_total, nrtb= self.nrtb, linear_modes= self.linear_modes,
            com=self.com, m_rigidR=self.m_rigidR, m_rigidT= self.m_rigidT, mapping= self.mapping, mapping_len = self.mapping_len)
        transformed_pdb.coords = new_coords
        return transformed_pdb

    def applyLinearTransformation(self, eta, pdb=None):
        """
        Apply slinear transformation
        :param eta: array of n_modes_total with the amplitudes
        :param pdb: PDB on which the transformation is applied
        :return:PDB
        """
        if pdb is None:
            pdb=self.pdb
        transformed_pdb = pdb.copy()
        for i in range(pdb.n_atoms):
            transformed_pdb.coords[i] = pdb.coords[i] + np.dot(eta, self.linear_modes[:,i])
        return transformed_pdb

@njit
def _fromAxisAngle(rkAxis, fRadians):
    fCos = np.cos(fRadians)
    fSin = np.sin(fRadians)
    fOneMinusCos = 1.0 - fCos
    fX2 = rkAxis[0] * rkAxis[0]
    fY2 = rkAxis[1] * rkAxis[1]
    fZ2 = rkAxis[2] * rkAxis[2]
    fXYM = rkAxis[0] * rkAxis[1] * fOneMinusCos
    fXZM = rkAxis[0] * rkAxis[2] * fOneMinusCos
    fYZM = rkAxis[1] * rkAxis[2] * fOneMinusCos
    fXSin = rkAxis[0] * fSin
    fYSin = rkAxis[1] * fSin
    fZSin = rkAxis[2] * fSin
    m = np.zeros((3, 3))
    m[0, 0] = fX2 * fOneMinusCos + fCos
    m[0, 1] = fXYM - fZSin
    m[0, 2] = fXZM + fYSin
    m[1, 0] = fXYM + fZSin
    m[1, 1] = fY2 * fOneMinusCos + fCos
    m[1, 2] = fYZM - fXSin
    m[2, 0] = fXZM - fYSin
    m[2, 1] = fYZM + fXSin
    m[2, 2] = fZ2 * fOneMinusCos + fCos
    return m


@njit
def applySpiralTransformation_njit(eta, oldPositions,
            nmodes, nmodes_total, nrtb, linear_modes, com, m_rigidR, m_rigidT, mapping,mapping_len):
    if len(eta) != nmodes_total:
        raise RuntimeError("Invalid number of normal mode amplitudes")

    translate_modes = 3
    EPSILON = 1e-11
    natoms = len(oldPositions)
    newPositions = np.zeros(oldPositions.shape, dtype=np.float32)
    oldVec = com.copy()
    oldPos = oldPositions.copy()

    amp = np.linalg.norm(eta)
    if (np.fabs(amp) < EPSILON):
        newPositions = oldPos
        return newPositions

    oldMat = np.zeros((nrtb, 3, 3))
    for i in range(nrtb):
        oldMat[i] = np.eye(3)

    # Big loop on modes; eta[mode] is the amplitude of the movement for this mode
    # The modes are ordered by their frequency, from the slowest (0) to the fastest (eta.size()-1). This is important. Null-space is excluded.
    for mode in range(nmodes_total):  # loop over modes
        if (eta[mode] == 0) or mode < translate_modes:
            continue

        # Define axes(3,ica) and trans(3,ica) from m_rigidR and m_rigidT(ica,imode)
        # This would be the main input of this subroutine
        axes = np.zeros((nrtb, 3))
        trans = np.zeros((nrtb, 3))
        for caId in range(nrtb):
            axes[caId] = m_rigidR[caId, mode] * eta[mode]
            trans[caId] = m_rigidT[caId, mode] * eta[mode]

        for caId in range(nrtb):
            # Define alpha, \vec{U} and \vec{T} as in Eq. (12) of your paper (Delta \Phi, \vec{n} and \vec{\Delta x}) from axes and trans read previously
            alpha = np.linalg.norm(axes[caId])
            u = axes[caId] / alpha
            T = trans[caId]
            # if False:
            # if mode <=translate_modes:
            if (np.fabs(alpha / amp) < EPSILON):
                for atomId in mapping[caId, :mapping_len[caId]]:
                    newPositions[atomId] = oldPos[atomId] + T
                    oldVec[caId] = oldVec[caId] + T
            else:
                # else, apply both Rot and Trans as in Eq. (17)
                # Calculate \Delta x colinear and orthogonal (Tcol and Tort), as in Eq. (14) of the paper
                # ‘|’ is the scalar product, and ‘^’ is the vector product (see below).

                u = np.dot(oldMat[caId], u)
                T = np.dot(oldMat[caId], T)
                Tcol = u * np.dot(T, u)
                Tort = T - Tcol

                Cold = oldVec[caId]
                X = Cold + np.cross(u, (
                            Tort / alpha))  # X is the center of rotation of this group, aka vec{r_0} of Eq. (16)

                alpha = np.fmod(alpha, 2.0 * np.pi)
                curMat = _fromAxisAngle(u, alpha)

                # Get rotation matrix R(Delta Phi, \vec n) from angle and axis. See the code below.
                mat = np.dot(curMat, oldMat[caId])

                # Get new COM (oldVec). This part is not yet documented in the manuscript.          // we project old onto new
                oldVec[caId] = np.dot(curMat, (oldVec[caId] - X)) + X + Tcol  # COM position

                # Apply Eq. (17), for each residue (RTB group)
                for atomId in mapping[caId, :mapping_len[caId]]:
                    rotCoord = np.dot(curMat, (oldPos[atomId] - X)) + X
                    rotCoord += Tcol
                    newPositions[atomId] = rotCoord
                oldMat[caId] = mat

        for i in range(natoms):
            oldPos[i] = newPositions[i]

        for i in range(translate_modes):
            newPositions += linear_modes[i] * eta[i]

    return newPositions
