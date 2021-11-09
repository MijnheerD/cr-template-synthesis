"""
Base class to provide read/write functionality in bunch, using internal directory structure.
"""

import os
import numpy as np
from contextlib import contextmanager


@contextmanager
def working_directory(path):
    """
    A context manager which changes the working directory to the given
    path, and then changes it back to its previous value on exit.
    Usage:
    > # Do something in original directory
    > with working_directory('/my/new/path'):
    >     # Do something in new directory
    > # Back to old directory
    """

    prev_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


class ReaderWriter(object):
    def __init__(self):
        self.filesX = []
        self.filesY = []

    def open_all_files(self, path):
        """
        Opens all files in a directory and stores their file handles.
        :param str path: Path to the directory
        """
        for file in os.listdir(os.path.join(path, 'fitX')):
            self.filesX.append(open(os.path.join(path, 'fitX', file)))
            self.filesY.append(open(os.path.join(path, 'fitY', file)))

    def close_all_files(self):
        """
        Close all the files opened by the instance.
        """
        for file in self.filesX:
            file.close()

        for file in self.filesY:
            file.close()


class ShowerDataC7(object):
    def __init__(self, sim_directory):
        self.name = os.path.basename(os.path.normpath(sim_directory)).split('_')[0]
        self.number = int(self.name[3:])

        self.particle_numbers = 0
        self.GH = []
        self.x_max = 0
        self.energy = 0
        self.type = None

        self.traces = None

        with working_directory(sim_directory):
            self.read_long()
            self.read_reas()
            self.read_time_traces()

    def read_long(self):
        long_file = f'../DAT{self.number}.long'

        long = np.genfromtxt(long_file, skip_header=2, skip_footer=216, usecols=(2, 3))
        hillas = np.genfromtxt(long_file, skip_header=422, skip_footer=2)

        self.particle_numbers = np.sum(long, axis=1)
        self.GH = hillas[2:]

    def read_reas(self):
        reas_file = f'../SIM{self.number}.reas'

        with open(reas_file, 'r') as file:
            for line in file:
                if 'DepthOfShowerMaximum' in line:
                    self.x_max = float(line.split()[2])
                elif 'PrimaryParticleEnergy' in line:
                    self.energy = float(line.split()[2])

    def read_time_traces(self, slice_thickness=5):
        from scipy.constants import c as c_vacuum

        bins_file = f'../SIM{self.number}_coreas.bins'
        signal_files = os.listdir('.')

        nr_of_antennas = len(np.unique(np.genfromtxt(bins_file)[:, 1]))
        nr_of_slices = len(signal_files)/nr_of_antennas
        nr_of_time_steps = len(np.genfromtxt(signal_files[0]))
        nr_of_pol = 3
        traces = np.zeros((nr_of_slices, nr_of_antennas, nr_of_time_steps, nr_of_pol))

        for file in signal_files:
            file_numbers = file.split('_')[1]
            antenna = int(file_numbers.split('x')[0])
            x_slice = int(file_numbers.split('x')[1])

            data = np.genfromtxt(file) * c_vacuum * 1e2
            traces[int(x_slice / 5 - 1), antenna, :, :] = data[:, 1:]

        self.traces = traces

    def get_sa_trace(self, x_slice, antenna):
        return self.traces[x_slice, antenna, :, :]
