"""
Read in all the values of the amplitude fit and perform a quadratic fit against Xmax
"""
import os.path

import numpy as np
from FileReaderWriter import ReaderWriter


class ParameterFitter(ReaderWriter):
    def __init__(self, directories):
        super().__init__()
        self.directories = directories
        self.particle_numbers = None
        self.x_max = None

    def open_files(self):
        for directory in self.directories:
            self.open_all_files(directory)

    def read_particle_numbers(self, long_directory):
        sim_nr = []
        for fileX in self.filesX:
            sim_nr.append(str(os.path.basename(fileX.name).split('.')[0].split('SIM')[1]))

        long_files = [os.path.join(long_directory, 'DAT' + nr + '.long') for nr in sim_nr
                      if os.path.isfile(os.path.join(long_directory, 'DAT' + nr + '.long'))]

        self.particle_numbers = np.array(
            [np.sum(np.genfromtxt(file, skip_header=2, skip_footer=216, usecols=(2, 3)), axis=1)
             for file in long_files])

        for file in long_files:
            with open(file, 'r') as f:
                # Skip the first lines, which contain the longitudinal evolution tables
                for _ in range(420):
                    next(f)
                # Search for the line containing the GH paramters
                line = f.readline()
                while 'PARAMETERS' not in line:
                    line = f.readline()
                # Extract Xmax from parameters
                self.x_max.append(float(line.split()[4]))

    def read_values(self):
        pass
