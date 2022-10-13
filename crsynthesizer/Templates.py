"""
TODO: extract distances from .bins file in TemplatesLibrary
"""

import os
import glob
import numpy as np
from FileReaderWriter import ShowerDataC7
from Fitters import AmplitudeFitter, ParameterFitter


class TemplatesLibrary(object):
    def __init__(self, dir_list: list):
        self.sim_list = []
        self.build_sim_list(dir_list)

        self.showers = [ShowerDataC7(shower) for shower in self.sim_list]
        self.distances = [1, 4000, 7500, 11000, 15000, 37500]  # Should be extractable from .bins file

    def build_sim_list(self, dir_list):
        for directory in dir_list:
            self.sim_list.extend(glob.glob(os.path.join(directory, '*_coreas/')))

    def get_primary_types(self):
        return [shower.type for shower in self.showers]

    def get_x_max(self):
        return [shower.x_max for shower in self.showers]

    def get_sa_ranges(self):
        antenna_range = self.showers[0].traces.shape[1]
        slices = self.showers[0].atm_slices
        return slices, list(range(antenna_range))


class TemplatesLibraryAnalyzer(object):
    def __init__(self, library: TemplatesLibrary, save_dir):
        self.library = library
        self.slices, self.antennas = library.get_sa_ranges()

        self.directory = save_dir

    def process_library(self, amp_fit_range=(1e7, 5e8), amp_fit_center=None, slice_range=None, antenna_range=None):
        if amp_fit_center is not None:
            assert amp_fit_range[0] <= amp_fit_center <= amp_fit_range[1], "fit_center must lie within fit_range"
        else:
            amp_fit_center = (amp_fit_range[1] + amp_fit_range[0]) / 2
        if slice_range is not None:
            assert max(slice_range) <= max(self.slices), f"Slice range exceeds biggest slice {max(self.slices)} g/cm^2"
        else:
            slice_range = self.slices
        if antenna_range is not None:
            assert max(antenna_range) <= max(self.antennas), f"There are only {max(self.antennas)} antennas"
        else:
            antenna_range = self.antennas

        for x_slice in slice_range:
            for antenna in antenna_range:
                d = max(0, 1e-9 * (x_slice / 400 - 1.5) * np.exp(1 - self.library.distances[antenna] / 40000))
                amp_fitters = [AmplitudeFitter(shower.get_sa_trace(x_slice, antenna), amp_fit_range,
                                               d ** 2, amp_fit_center).fit_amplitude_iter()
                               for shower in self.library.showers]
                param_fitter = ParameterFitter(np.array([fitter.opt for fitter in amp_fitters]),
                                               np.array([fitter.cov for fitter in amp_fitters]),
                                               self.library.get_x_max()).fit_parameters()

                param_fitter.save_to_file(os.path.join(self.directory, f'param_{antenna}x{int(x_slice)}'))
