import os
import glob
from FileReaderWriter import ShowerDataC7


class TemplatesLibrary(object):
    def __init__(self, dir_list):
        self.sim_list = []
        self.build_sim_list(dir_list)

        self.showers = [ShowerDataC7(shower) for shower in self.sim_list]

    def build_sim_list(self, dir_list):
        for directory in dir_list:
            self.sim_list.extend(glob.glob(os.path.join(directory, '*_coreas/')))

    def get_primary_types(self):
        return [shower.type for shower in self.showers]

    def get_x_max(self):
        return [shower.x_max for shower in self.showers]
