from afmfit.utils import euler2matrix

import numpy as np
import os
import copy
from Bio.SVDSuperimposer import SVDSuperimposer
import tempfile
from afmfit.utils import run_chimerax

class PDB:

    @classmethod
    def read_coords(cls, pdb_file):
        """
        Read coords only
        :param pdb_file: PDB file
        :return: array of n_atoms*3
        """
        coords = []
        with open(pdb_file, "r") as f:
            for line in f:
                if 'ATOM' in line:
                    coords.append([
                        line[30:38], line[38:46], line[46:54]
                    ])
        return np.array(coords).astype(float)

    def __init__(self, pdb_file):
        """
        PDB object
        :param pdb_file: pdb file
        """
        if not os.path.exists(pdb_file):
            raise RuntimeError("Could not read PDB file : No such file")
        if os.path.getsize(pdb_file) == 0:
            raise RuntimeError("Could not read PDB file : file empty")

        atom = []
        atomNum = []
        atomName = []
        resName = []
        resAlter = []
        chainName = []
        resNum = []
        insertion = []
        coords = []
        occ = []
        temp = []
        chainID = []
        elemName = []
        with open(pdb_file, "r") as f:
            for line in f:
                spl = line.split()
                if len(spl) > 0:
                    if (spl[0] == 'ATOM'):  # or (hetatm and spl[0] == 'HETATM'):
                        l = [line[:6], line[6:11], line[12:16], line[16], line[17:21], line[21], line[22:26],
                             line[26], line[30:38],
                             line[38:46], line[46:54], line[54:60], line[60:66], line[72:76], line[76:78]]
                        l = [i.strip() for i in l]
                        atom.append(l[0])
                        atomNum.append(l[1])
                        atomName.append(l[2])
                        resAlter.append(l[3])
                        resName.append(l[4])
                        chainName.append(l[5])
                        resNum.append(l[6])
                        insertion.append(l[7])
                        coords.append([float(l[8]), float(l[9]), float(l[10])])
                        occ.append(l[11])
                        temp.append(l[12])
                        chainID.append(l[13])
                        elemName.append(l[14])

        atomNum = np.array(atomNum)
        atomNum[np.where(atomNum == "*****")[0]] = "-1"

        self.atom = np.array(atom, dtype='<U6')
        self.n_atoms = len(self.atom)
        self.atomNum = np.array(atomNum).astype(int)
        self.atomName = np.array(atomName, dtype='<U4')
        self.resName = np.array(resName, dtype='<U4')
        self.resAlter = np.array(resAlter, dtype='<U1')
        self.chainName = np.array(chainName, dtype='<U1')
        self.resNum = np.array(resNum).astype(int)
        self.insertion = np.array(insertion, dtype='<U1')
        self.coords = np.array(coords).astype(np.float32)
        self.occ = np.array(occ).astype(np.float32)
        self.temp = np.array(temp).astype(np.float32)
        self.chainID = np.array(chainID, dtype='<U4')
        self.elemName = np.array(elemName, dtype='<U2')

        self.active_atoms = np.arange(self.n_atoms)

        if self.n_atoms == 0:
            raise RuntimeError("Could not read PDB file : PDB file is empty")

    def viewChimera(self):
        """
        Show PDB in ChimeraX
        """
        with tempfile.TemporaryDirectory() as tmdir:
            fname = os.path.join(tmdir,"pdb.pdb")
            self.write_pdb(fname)
            run_chimerax(fname)

    def write_pdb(self, file):
        """
        Write to PDB Format
        :param file: pdb file path
        """
        with open(file, "w") as file:
            past_chainName = self.chainName[0]
            past_chainID = self.chainID[0]
            for i in range(len(self.atom)):
                if past_chainName != self.chainName[i] or past_chainID != self.chainID[i]:
                    past_chainName = self.chainName[i]
                    past_chainID = self.chainID[i]
                    file.write("TER\n")

                atom = self.atom[i].ljust(6)  # atom#6s
                if self.atomNum[i] == -1 or self.atomNum[i] >= 100000:
                    atomNum = "99999"  # aomnum#5d
                else:
                    atomNum = str(self.atomNum[i]).rjust(5)  # aomnum#5d
                if len(self.atomName[i]) >= 4:
                    atomName = self.atomName[i].ljust(4)
                else:
                    atomName = " " + self.atomName[i].ljust(3)
                resAlter = self.resAlter[i].ljust(1)  # resAlter#1
                resName = self.resName[i].ljust(4)  # resname#1s
                chainName = self.chainName[i].rjust(1)  # Astring
                resNum = str(self.resNum[i]).rjust(4)  # resnum
                insertion = self.insertion[i].ljust(1)  # resnum
                coordx = str('%8.3f' % (float(self.coords[i][0]))).rjust(8)  # x
                coordy = str('%8.3f' % (float(self.coords[i][1]))).rjust(8)  # y
                coordz = str('%8.3f' % (float(self.coords[i][2]))).rjust(8)  # z\
                occ = str('%6.2f' % self.occ[i]).rjust(6)  # occ
                temp = str('%6.2f' % self.temp[i]).rjust(6)  # temp
                chainID = str(self.chainID[i]).ljust(4)  # elname
                elemName = str(self.elemName[i]).rjust(2)  # elname
                file.write("%s%s %s%s%s%s%s%s   %s%s%s%s%s      %s%s\n" % (
                    atom, atomNum, atomName, resAlter, resName, chainName, resNum,
                    insertion, coordx, coordy, coordz, occ, temp, chainID, elemName))
            file.write("END\n")

    def matchPDBatoms(self, reference_pdb, ca_only=False, matchingType=None):
        """
        match atoms between the pdb and a reference pdb
        :param reference_pdb: PDB
        :param ca_only: True if carbon alph only
        :param matchingType: 0= chain first, 1= segment ID first
        :return: index of matching atoms
        """
        print("> Matching PDBs atoms ...")
        n_mols = 2

        if matchingType == None:
            chain_name_list1 = self.get_chain_list(chainType=0)
            chain_name_list2 = reference_pdb.get_chain_list(chainType=0)
            n_matching_chain_names = sum([i in chain_name_list2 for i in chain_name_list1])
            print("Chains list 1  : " + str(chain_name_list1))
            print("Chains list 1  : " + str(chain_name_list2))
            print("Number of match : " + str(n_matching_chain_names))

            chain_id_list1 = self.get_chain_list(chainType=1)
            chain_id_list2 = reference_pdb.get_chain_list(chainType=1)
            n_matching_chain_ids = sum([i in chain_id_list2 for i in chain_id_list1])
            print("Seg list 1  : " + str(chain_id_list1))
            print("Seg list 1  : " + str(chain_id_list1))
            print("Number of match : " + str(n_matching_chain_ids))

            if n_matching_chain_ids == 0 and n_matching_chain_names == 0:
                raise RuntimeError("No matching chains")
            elif n_matching_chain_ids >= n_matching_chain_names:
                matchingType = 1
                print("\t Matching segments %s ... " % n_matching_chain_ids)
            elif n_matching_chain_ids < n_matching_chain_names:
                matchingType = 0
                print("\t Matching chains %s ... " % n_matching_chain_names)

        ids = []
        ids_idx = []
        for m in [self, reference_pdb]:
            id_tmp = []
            id_idx_tmp = []
            for i in range(m.n_atoms):
                if (not ca_only) or m.atomName[i] == "CA" or m.atomName[i] == "P":
                    id_tmp.append("%s_%i_%s_%s" % (m.chainName[i] if matchingType == 0 else m.chainID[i],
                                                   m.resNum[i], m.resName[i], m.atomName[i]))
                    id_idx_tmp.append(i)
            ids.append(np.array(id_tmp))
            ids_idx.append(np.array(id_idx_tmp))

        idx = []
        for i in range(len(ids[0])):
            idx_line = [ids_idx[0][i]]
            for m in range(1, n_mols):
                idx_tmp = np.where(ids[0][i] == ids[m])[0]
                if len(idx_tmp) == 1:
                    idx_line.append(ids_idx[m][idx_tmp[0]])
                elif len(idx_tmp) > 1:
                    print("\t Warning : One atom in mol#0 is matching several atoms in mol#%i : " % m)

            if len(idx_line) == n_mols:
                idx.append(idx_line)

        if len(idx) == 0:
            print("\t Warning : No matching coordinates")

        print("\t %i matching atoms " % len(np.array(idx)))
        print("\t Done")

        return np.array(idx)

    def set_active_atoms(self, idx):
        self.active_atoms = self.active_atoms[idx]
    def set_active_atoms_ca(self):
        self.active_atoms = self.active_atoms[self.allatoms2ca()]

    def alignMol(self, reference_pdb, idx_matching_atoms=None):
        """
        Align PDB with another PDB using BioPython
        :param reference_pdb: reference (fix) PDB
        :param idx_matching_atoms: Indexes of atoms that matches if PDBs are different
        :return: Copy of PDB aligned to the reference
        """
        # print("> Aligning PDB ...")

        if idx_matching_atoms is not None:
            c1 = reference_pdb.coords[idx_matching_atoms[:, 1]]
            c2 = self.coords[idx_matching_atoms[:, 0]]
        else:
            c1 = reference_pdb.coords
            c2 = self.coords

        rot, tran = self.alignCoords(c1, c2)
        self_copy = self.copy()
        self_copy.coords = np.dot(self_copy.coords, rot) + tran
        # print("\t Done \n")

        return self_copy

    @classmethod
    def alignCoords(cls, coord_ref, coord):
        """
        Align set of coordinates
        :param coord_ref: reference (fix) coordinates
        :param coord: coordinates to fit
        :return: aligned coordinates
        """
        sup = SVDSuperimposer()
        sup.set(coord_ref, coord)
        try:
            sup.run()
            rot, tran = sup.get_rotran()
        except np.linalg.LinAlgError:
            print("Error while aligning : SVD did not converge")
            rot = np.eye(3)
            tran = np.zeros(3)
        return rot, tran

    def rotate(self, angles):
        """
        Rotate the coordinates
        :param angles: list of 3 Euler angles
        """
        R= euler2matrix(angles)
        self.coords = np.dot( np.array(R,np.float32), self.coords.T).T

    def translate(self, shift):
        """
        Translate the coordinates
        :param shift: list of 3 shifts
        """
        self.coords += np.array(shift,np.float32)

    def get_radius(self):
        """
        Radius of the atoms
        :return: radii
        """
        radius = {
            "H": 1.2,
            "O": 1.5,
            "S": 1.6,
            "N": 1.5,
            "C": 1.7
        }
        r = []
        for i in range(self.n_atoms):
            ename = self.elemName[i]
            if ename not in radius:
                raise RuntimeError("Unknown element : " + ename)
            r.append(radius[ename])
        return np.array(r)

    def getRMSD(self, reference_pdb, align=False, idx_matching_atoms=None):
        """
        Claculate RMSD with another PDB
        :param reference_pdb: other PDB
        :param align: True to align the PDBs before calculating the RMSD
        :param idx_matching_atoms: Indexes of atoms that matches if PDBs are different
        :return: RMSD
        """
        if align:
            aligned = self.alignMol(reference_pdb=reference_pdb, idx_matching_atoms=idx_matching_atoms)
        else:
            aligned = self
        if idx_matching_atoms is not None:
            coord1 = reference_pdb.coords[idx_matching_atoms[:, 1]]
            coord2 = aligned.coords[idx_matching_atoms[:, 0]]
        else:
            coord1 = reference_pdb.coords
            coord2 = aligned.coords
        return np.sqrt(np.mean(np.square(np.linalg.norm(coord1 - coord2, axis=1))))

    def select_atoms(self, idx):
        """
        Select atoms in the PDB
        :param idx: Indexes of the atoms to select
        """
        self.coords = self.coords[idx]
        self.n_atoms = self.coords.shape[0]
        self.atom = self.atom[idx]
        self.atomNum = self.atomNum[idx]
        self.atomName = self.atomName[idx]
        self.resName = self.resName[idx]
        self.resAlter = self.resAlter[idx]
        self.chainName = self.chainName[idx]
        self.resNum = self.resNum[idx]
        self.insertion = self.insertion[idx]
        self.elemName = self.elemName[idx]
        self.occ = self.occ[idx]
        self.temp = self.temp[idx]
        self.chainID = self.chainID[idx]
        self.active_atoms = np.arange(self.n_atoms)

    def get_chain_list(self, chainType=0):
        if chainType == 0:
            lst = list(set(self.chainName))
        else:
            lst = list(set(self.chainID))
        lst.sort()
        return lst

    def get_chain(self, chainName):
        if not isinstance(chainName, list):
            chainName = [chainName]
        chainidx = []
        for i in chainName:
            idx = np.where(self.chainName == i)[0]
            if len(idx) == 0:
                idx = np.where(self.chainID == i)[0]
            chainidx = chainidx + list(idx)
        return np.array(chainidx)

    def select_chain(self, chainName):
        self.select_atoms(self.get_chain(chainName))

    def copy(self):
        return copy.deepcopy(self)

    def allatoms2ca(self):
        new_idx = []
        for i in range(self.n_atoms):
            if self.atomName[i] == "CA" or self.atomName[i] == "P":
                new_idx.append(i)
        return np.array(new_idx)

    def center(self):
        """
        Center the coordinates
        """
        self.coords -= np.mean(self.coords, axis=0)

def dcd2numpyArr(filename):
    """
    Read coordinate file (.DCD)
    :param filename: DCD file
    :return: coordinates ncoord * n_atoms * 3
    """
    print("> Reading dcd file %s" % filename)
    BYTESIZE = 4
    with open(filename, 'rb') as f:

        # Header
        # ---------------- INIT

        start_size = int.from_bytes((f.read(BYTESIZE)), "little")
        crd_type = f.read(BYTESIZE).decode('ascii')
        nframe = int.from_bytes((f.read(BYTESIZE)), "little")
        start_frame = int.from_bytes((f.read(BYTESIZE)), "little")
        len_frame = int.from_bytes((f.read(BYTESIZE)), "little")
        len_total = int.from_bytes((f.read(BYTESIZE)), "little")
        for i in range(5):
            f.read(BYTESIZE)
        time_step = np.frombuffer(f.read(BYTESIZE), dtype=np.float32)
        for i in range(9):
            f.read(BYTESIZE)
        charmm_version = int.from_bytes((f.read(BYTESIZE)), "little")

        end_size = int.from_bytes((f.read(BYTESIZE)), "little")

        if end_size != start_size:
            raise RuntimeError("Can not read dcd file")

        # ---------------- TITLE
        start_size = int.from_bytes((f.read(BYTESIZE)), "little")
        ntitle = int.from_bytes((f.read(BYTESIZE)), "little")
        tilte_rd = f.read(BYTESIZE * 20 * ntitle)
        try:
            title = tilte_rd.encode("ascii")
        except AttributeError:
            title = str(tilte_rd)
        end_size = int.from_bytes((f.read(BYTESIZE)), "little")

        if end_size != start_size:
            raise RuntimeError("Can not read dcd file")

        # ---------------- NATOM
        start_size = int.from_bytes((f.read(BYTESIZE)), "little")
        natom = int.from_bytes((f.read(BYTESIZE)), "little")
        end_size = int.from_bytes((f.read(BYTESIZE)), "little")

        if end_size != start_size:
            raise RuntimeError("Can not read dcd file")

        # ----------------- DCD COORD
        dcd_arr = np.zeros((nframe, natom, 3), dtype=np.float32)
        for i in range(nframe):
            for j in range(3):

                start_size = int.from_bytes((f.read(BYTESIZE)), "little")
                while (start_size != BYTESIZE * natom and start_size != 0):
                    # print("\n-- UNKNOWN %s -- " % start_size)

                    f.read(start_size)
                    end_size = int.from_bytes((f.read(BYTESIZE)), "little")
                    if end_size != start_size:
                        raise RuntimeError("Can not read dcd file")
                    start_size = int.from_bytes((f.read(BYTESIZE)), "little")

                bin_arr = f.read(BYTESIZE * natom)
                if len(bin_arr) == BYTESIZE * natom:
                    dcd_arr[i, :, j] = np.frombuffer(bin_arr, dtype=np.float32)
                else:
                    break
                end_size = int.from_bytes((f.read(BYTESIZE)), "little")
                if end_size != start_size:
                    if i > 1:
                        break
                    else:
                        # pass
                        raise RuntimeError("Can not read dcd file %i %i " % (start_size, end_size))

        print("\t -- Summary of DCD file -- ")
        print("\t\t crd_type  : %s" % crd_type)
        print("\t\t nframe  : %s" % nframe)
        print("\t\t len_frame  : %s" % len_frame)
        print("\t\t len_total  : %s" % len_total)
        print("\t\t time_step  : %s" % time_step)
        print("\t\t charmm_version  : %s" % charmm_version)
        print("\t\t title  : %s" % title)
        print("\t\t natom  : %s" % natom)
    print("\t Done \n")

    return dcd_arr

def numpyArr2dcd(arr, filename, start_frame=1, len_frame=1, time_step=1.0, title=None):
    """
    Write coordinate file (.DCD)
    :param arr: coordinates ncoord * natoms * 3
    :param filename: DCD file
    :param start_frame:
    :param len_frame:
    :param time_step:
    :param title:
    """
    print("> Wrinting dcd file %s" % filename)
    BYTESIZE = 4
    nframe, natom, _ = arr.shape
    len_total = nframe * len_frame
    charmm_version = 24
    if title is None:
        title = "DCD file generated by AFMfit"
    ntitle = (len(title) // (20 * BYTESIZE)) + 1
    with open(filename, 'wb') as f:
        zeroByte = int.to_bytes(0, BYTESIZE, "little")

        # Header
        # ---------------- INIT
        f.write(int.to_bytes(21 * BYTESIZE, BYTESIZE, "little"))
        f.write(b'CORD')
        f.write(int.to_bytes(nframe, BYTESIZE, "little"))
        f.write(int.to_bytes(start_frame, BYTESIZE, "little"))
        f.write(int.to_bytes(len_frame, BYTESIZE, "little"))
        f.write(int.to_bytes(len_total, BYTESIZE, "little"))
        for i in range(5):
            f.write(zeroByte)
        f.write(np.float32(time_step).tobytes())
        for i in range(9):
            f.write(zeroByte)
        f.write(int.to_bytes(charmm_version, BYTESIZE, "little"))

        f.write(int.to_bytes(21 * BYTESIZE, BYTESIZE, "little"))

        # ---------------- TITLE
        f.write(int.to_bytes((ntitle * 20 + 1) * BYTESIZE, BYTESIZE, "little"))
        f.write(int.to_bytes(ntitle, BYTESIZE, "little"))
        f.write(title.ljust(20 * BYTESIZE).encode("ascii"))
        f.write(int.to_bytes((ntitle * 20 + 1) * BYTESIZE, BYTESIZE, "little"))

        # ---------------- NATOM
        f.write(int.to_bytes(BYTESIZE, BYTESIZE, "little"))
        f.write(int.to_bytes(natom, BYTESIZE, "little"))
        f.write(int.to_bytes(BYTESIZE, BYTESIZE, "little"))

        # ----------------- DCD COORD
        for i in range(nframe):
            for j in range(3):
                f.write(int.to_bytes(BYTESIZE * natom, BYTESIZE, "little"))
                f.write(np.float32(arr[i, :, j]).tobytes())
                f.write(int.to_bytes(BYTESIZE * natom, BYTESIZE, "little"))
    print("\t Done \n")




