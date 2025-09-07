import numpy as np
import glob
import mdtraj as md
import os
import pandas as pd
from shutil import which
import subprocess

## helper functions
def mkdir( directory ):
    """os.mkdir returns an exception if a directory exists. mkdir checks if a directory exists before creating it."""
    if not os.path.isdir(directory):
        os.mkdir( directory )

def detect_unfolding(arr, high_cutoff, low_cutoff, eps):
    """Detects all the transitions from high_cutoff to low_cutoff occurring in an array.
    Returns an list with the initial and final indices of all the transitions."""
    dim = len(arr)
    time_high = 0
    paths = []
    time = 0
    while (time < dim):
        item = arr[time]
        if (item < high_cutoff + eps and item > high_cutoff - eps):
            time_high = time
            counter = 1
            for jtem in arr[time_high + 1:]:
                if (jtem < high_cutoff + eps and jtem > high_cutoff - eps):
                    time = time_high + counter - 1
                    break
                elif (jtem < low_cutoff + eps and jtem > low_cutoff - eps):
                    paths.append([time_high, time_high + counter])
                    time = time_high + counter - 1
                    break
                counter += 1
        time += 1
    return paths

def quick_load(file):
    """Uses pandas read_csv to read a table file. Much faster than numpy.genfromtxt() and numpy.loadtxt()."""
    dataset = pd.read_csv(file, delimiter=r"\s+", header=None, comment='#')
    return dataset.values

def run_process( process ):
    """Runs bash command"""
    p = subprocess.Popen(process, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         universal_newlines=True)
    p.communicate()


class NotExecutable(Exception):
    """Custom exception. Used to raise a NotExecutable error when a program is recognized as not executable."""
    pass

## main class
class Go2SAXS():
    def __init__(self, sim='trajectories', top='top.pdb', pulchra='pulchra', pepsi='Pepsi-SAXS', plumed='plumed',
                  maxc=0.9, minc=0.6, stride=10, err_scale=1, pepsi_params=[], plumedf='plumed.dat', exp_ref='exp.dat'):
        """
        Class constructor

        Input
        ----
        sim (str): path to the directory with simulations
        top (str): path to the topology file
        pulchra (str): path to the Pulchra executable
        pepsi (str): path to Pepsi-SAXS executable
        plumed (str): path to Plumed executable
        maxc (float): high-cutoff in the CV to detect unfolding
        minc (float): low-cutoff in the CV to detect unfolding
        stride (int): stride to load trajectories
        err_scale (float): scaling factor for experimental errors
        pepsi_params (list): list [I0, drho, r_0] parameters to pass to Pepsi-SAXS
        plumedf (str): path to Plumed file to compute CV. Do not specify a directory in the output file.
        exp_ref (str): path to reference experimental file
        """

        # path to the directoriy where where synthetic experimental trajectories are located
        self.exp_sim = sim + '/'
        # path to the directory where to store the observables associated to the trajectories
        self.exp_obs = self.exp_sim + 'obs/'
        # path to the directory where to store single frames for Pulchra side-chain re-assignement
        self.exp_str = self.exp_sim + 'structs/'
        # path to the directory where to store single frames for Pulchra side-chain re-assignement
        self.exp_int = self.exp_sim + 'intensities/'
        # path to the directory where resampled intensities are saved
        self.av_int = self.exp_sim + 'average_intensities/'
        # path to the directory where resampled intensities are saved
        self.resamp = self.exp_sim + 'resampled_intensities/'

        # create directories
        mkdir(self.exp_obs)
        mkdir(self.exp_str)
        mkdir(self.exp_int)
        mkdir(self.resamp)
        mkdir(self.av_int)

        # check the simulation directory exists and it has simulations in it
        if os.path.isdir(self.exp_sim):
            self.ntrajs = len(glob.glob(self.exp_sim + '*.xtc'))
            assert self.ntrajs != 0, f'No xtc trajectories found in {self.exp_sim}.'
        else:
            raise NameError(f'The simulation directory {self.exp_sim} does not exist.')

        # locations of Pulchra, PEPSI-SAXS and PLUMED
        self.pulchra = pulchra
        self.pepsi = pepsi
        self.plumed = plumed

        # check the paths to Pulchra, Pepsi and Plumed all link to executables
        if which(pulchra) is None:
            raise NotExecutable(f'{pulchra} is not executable or does not exist.')
        if which(pepsi) is None:
            raise NotExecutable(f'{pepsi} is not executable or does not exist.')
        if which(plumed) is None:
            raise NotExecutable(f'{plumed} is not executable or does not exist.')

        # cutoffs needed to detect unfoldings in MD simulations
        self.max_cut = maxc
        self.min_cut = minc
        self.eps = 0.03

        # stride to read trajectories and topology to read them
        self.stride = int(stride)
        self.top = top

        # check the stride is non-zero and that the topology exists
        assert stride != 0, "Trajectory stride cannot be 0."
        if not os.path.isfile(self.top):
            raise NameError(f'The topology file {self.top} does not exist.')

        # plumed file to compute fraction of native contacts
        self.plumedf = plumedf

        # reference experimental file for q-values and errors
        self.exp_ref = exp_ref
        self.err_scale = err_scale

        # parameters to pass Pepsi
        assert len(pepsi_params) == 3, "The pepsi_params list should contain I0, drho and r_0."
        self.I0 = pepsi_params[0]
        self.drho = pepsi_params[1]
        self.r0 = pepsi_params[2]

        # load and modify the reference experimental file
        exp = quick_load(self.exp_ref)
        self.q = np.copy(exp[:,0])
        self.err = np.copy(exp[:,2])
        exp[:,1].fill(1)
        exp[:,2].fill(1)
        np.savetxt(self.exp_sim + "exp.dat", exp) # needed for Pepsi-SAXS

        # list where to store experimental intensities
        self.intensity = []

    def compute_CV(self, cvname='Q', trajname='md_'):
        """Runs Plumed to compute the CV of interest

        Input
        -----
        cvname (str): global name of CV files
        trajname (str): global name of trajectories
        """

        tmp = len(glob.glob(self.exp_obs + f"/{cvname}*"))
        if tmp == self.ntrajs:
            print("# Fraction of native contacts already computed. Skipping.")
        else:
            for i in range(self.ntrajs):
                print("# Computing CV: {i}/{n}\r".format(i=i, n=self.ntrajs), end = "")
                # Run plumed
                process = f"{self.plumed} driver --mf_xtc {self.exp_sim}/{trajname}{i}.xtc --plumed {self.plumedf}"
                run_process(process)

                # Move the file in a new directory
                process = f"mv {cvname} {self.exp_obs}/{cvname}{i}"
                run_process(process)
            tmp = len(glob.glob(self.exp_obs + f"/{cvname}*"))
            assert tmp == self.ntrajs, f"There has been an error with the calculation of CVs: {tmp} =/= {self.ntrajs}"
        self.computed_cv = True

    def reassign_sidechains(self, cvname='Q', trajname='md_', check_unfolding=True):

        """Reassigns side-chains to Go-model structures using Pulchra

        Input
        -----
        cvname (str): global name of CV files
        trajname (str): global name of trajectories
        check_unfolding (bool): check for unfolding in the simulation based on the array of computed CVs
        """

        k = 0
        for i in range(self.ntrajs):
            if check_unfolding:
                assert self.computed_cv, "You have not computed CVs from the simulations. Run compute_CV() first."
                q = quick_load(f'{self.exp_obs}/{cvname}{i}')[:,1] #assumes here that Plumed file is unmodified
                path = detect_unfolding(q, self.max_cut, self.min_cut, self.eps)
            else:
                path = [0]

            if path != []:
                traj = md.load(f'{self.exp_sim}/{trajname}{i}.xtc', top=self.top, stride=self.stride)
                l = len(traj)

                sdir = f"{self.exp_str}/{i}/"
                mkdir(sdir)
                for f,t in enumerate(traj):
                    print("# Re-assigning sidechains using Pulchra: {i}/{n}\r".format(i=k, n=self.ntrajs*l), end="")
                    t.save_pdb(f'{sdir}{f}.pdb')
                    process = f"{self.pulchra} {sdir}{f}.pdb"
                    run_process(process)
                    k += 1

    def compute_intensity(self):
        """Computes SAXS intensities from simulations using Pepsi-SAXS"""
        k = 0
        trajs = glob.glob(f'{self.exp_str}/*')
        assert trajs, f"No structures found in {self.exp_str}. Run reassign_sidechains() first."
        for n,t in enumerate(trajs):
            nframes = len( glob.glob( t + '/*rebuilt.pdb' ) )
            out_int = f"{self.exp_int}/{n}/"
            mkdir(out_int)
            for i in range(nframes):
                print("# Computing SAXS intensities: {i}/{n}\r".format(i=k, n=self.ntrajs * nframes), end="")
                process = f"{self.pepsi} {t}/{i}.pdb {self.exp_sim}/exp.dat -o {out_int}/{i}.dat -cst --cstFactor 0 " \
                          f"--I0 {self.I0} --dro {self.drho} --r0_min_factor {self.r0} " \
                          f"--r0_max_factor {self.r0} --r0_N 1"
                run_process(process)
                k += 1

    def average_intensities(self):
        """Averages the intensities among simulations to get synthetic experimental data"""
        trajs = glob.glob(f'{self.exp_int}/*')
        assert trajs, f"No intensities found in {self.exp_int}. Run compute_intensities() first."
        nint = len(glob.glob(f'{self.exp_int}/0/*.dat'))
        header = '#  DATA=SAXS PRIOR=GAUSS' # header to be interpreted by BME
        intensity = []
        for i in range(nint):
            print("# Averaging intensities: {i}/{n}\r".format(i=i, n=nint), end="")
            av = []
            for t in trajs:
                av.append(quick_load(f'{t}/{i}.dat')[:,3])
            tmp = np.average(av, axis=0)
            ii = np.empty((len(tmp), 3))
            ii[:,0] = self.q
            ii[:,1] = tmp
            ii[:,2] = self.err
            intensity.append(tmp)
            np.savetxt(f'{self.av_int}/intensity_{i}.dat', ii, fmt='%g', header=header)
        self.intensity = np.array(intensity)

    def resample_intensities(self):
        """Adds noise to the averaged intensities based on the experimental error"""
        header = '  DATA=SAXS PRIOR=GAUSS'  # header to be interpreted by BME
        if not self.intensity:
            assert glob.glob(f'{self.av_int}/*'), f"No average intensities found in {self.av_int}. " \
                                                  f"Run average_intensities() first."
            tmp = quick_load(f'{self.av_int}/intensity_{i}.dat')[:,1]
            self.intensity.append(tmp)
        for n,i in enumerate(self.intensity):
            intensity = np.empty((len(self.q), 3))
            intensity[:,0] = self.q
            intensity[:,2] = self.err_scale * self.err
            intensity[:,1] = np.random.normal(i, intensity[:,2])
            np.savetxt(f"{self.resamp}/intensity_{n}.dat", intensity, fmt='%g', header=header)

    def SAXS2BME(self):
        """Converts intensities into a file that can be fed to BME"""
        trajs = glob.glob(f'{self.exp_int}/*/')
        nint = len(glob.glob(f'{self.exp_int}/0/*.dat'))
        Is_sim = self.q
        print("# Building BME file ...")
        for t in trajs: # for every simulation (0)
            for i in range(nint): # for every frame in each simulation
                I_f = np.loadtxt(f'{t}/{i}.dat')[:,3]
                Is_sim = np.vstack((Is_sim, I_f.T))
            index = np.arange(0, nint, 1).reshape(nint, 1)
            Is_sim = np.concatenate((index, Is_sim[1:,:]), axis=1)  # removes first line (qs), and adds index column
            header_arr = 'label ' + ' '.join(map(str, self.q))
            np.savetxt(f'{t}/BME.dat', Is_sim, delimiter=' ', header=header_arr)

def main():
    sim = '' #
    top = 'data/bsa_ca.pdb' #
    pulchra = '/storage1/francesco/software/pulchra'
    pepsi = '/storage1/francesco/software/Pepsi-SAXS'
    plumed = '/storage1/francesco/software/plumed-2.5.4/bin/plumed'
    plumedf = 'data/plumed.dat'
    exp_ref = 'data/exp.dat'
    stride = 1 #
    err_scale = 1 #
    params = [0.01, 1., 1.025] #optimized by Francesco for IDPs #
    unfolding_sims = False # the simulations are unfolding simulations (True) or metad ones (False)
    cvname = 'Q'
    trajname = 'md_'

    g2s = Go2SAXS(sim=sim, top=top, pulchra=pulchra, pepsi=pepsi, plumed=plumed, stride=stride, plumedf=plumedf,
                  exp_ref=exp_ref,err_scale=err_scale, pepsi_params=params)
    if unfolding_sims:
        g2s.compute_CV(cvname=cvname, trajname=trajname)
        g2s.reassign_sidechains(cvname=cvname, trajname=trajname, check_unfolding=True)
        g2s.compute_intensity()
        g2s.average_intensities()
        g2s.resample_intensities()
    else:
        g2s.reassign_sidechains(check_unfolding=False)
        g2s.compute_intensity()
        g2s.SAXS2BME()

if __name__ == "__main__":
    main()
