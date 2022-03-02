import os
import glob
import numpy as np
from scipy.optimize import fmin
from NumpyEncoder import write_file_json
from crsynthesizer.FileReaderWriter import working_directory
from GlobalVars import DATABASE_DIRECTORY, FIT_DIRECTORY, REAS_DIRECTORY


def shape_r_l(p, x):
    x1 = x - p[2]
    fn = np.exp(-x1 / (p[0] * p[1])) * np.power((1 + p[0] * x1 / p[1]), 1 / (p[0] * p[0]))

    return fn


def residual_mse(p, x, y):
    res = np.sum((y - shape_r_l(p, x)) ** 2)

    return res


def fit_r_l(atm_depth, n_tot):  # longitudinal profile :slanted depth vs # particle(em/charged)
    n1 = n_tot / np.max(n_tot)
    int_rl = np.array([0.3, 210, 620])
    parfit = fmin(residual_mse, int_rl.tolist(), args=(np.array(atm_depth), n1), disp=False)

    return parfit


def read_long(file):
    long = np.genfromtxt(file, skip_header=2, skip_footer=216, usecols=(0, 2, 3))
    return long[:, 0], np.sum(long[:, 1:], axis=1)


def read_g_h_fit(file):
    long = np.genfromtxt(file, skip_header=422, skip_footer=2, usecols=(2, 3, 4))
    return long


def sims_r_l(directory):
    """
    Construct a dictionary with the simulation numbers as keys and a tuple of R, L and X_max as value.
    :param directory: Directory containing the .long files of the simulations.
    :return: The dictionary with R, L and X_max per simulation.
    """
    sims_dict = {}

    with working_directory(directory):
        long_files = glob.glob('DAT*.long')

        for file in long_files:
            nr = file.split('.')[0][3:]
            x_slices, n_charged = read_long(file)

            sims_dict[nr] = fit_r_l(x_slices[10:-10], n_charged[10:-10])  # R, L, X_max

    return sims_dict


FIT_FILES = ['fitFilesUB17', 'fitFilesUB18', 'fitFilesUB19']
FIT_DIRS = [os.path.join(FIT_DIRECTORY, file) for file in FIT_FILES]

simsRL = sims_r_l(REAS_DIRECTORY)
mass = {'proton': 938.272, 'iron': 26 * 931.494}

database = []
for DIR in FIT_DIRS:
    for file in os.listdir(os.path.join(DIR, 'fitX')):
        sim = os.path.basename(file)
        sim_nr = sim.split('.')[0][3:]

        n_max, depth, x_max = read_g_h_fit(os.path.join(REAS_DIRECTORY, f'DAT{sim_nr}.long'))
        r, l, x_max_rl = simsRL[sim_nr]
        m = mass['proton'] if sim_nr[0] == '1' else mass['iron']

        database.append([sim_nr, x_max_rl, n_max, l, r, depth, m])

write_file_json(os.path.join(DATABASE_DIRECTORY, 'sim_parameters.json'), database)
