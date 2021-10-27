import os
import numpy as np
from scipy.constants import c as c_vacuum

SIM_DIRECTORY = '/mnt/hgfs/Shared data/BulkSynth/bulksynth-17/'
PARAM_DIRECTORY = '/home/mdesmet/PycharmProjects/cr-template-synthesis/Amplitude spectrum fitting/paramFitCenter/'
TEMPLATE_NR = '100052'


def get_number_of_particles(path):
    """
    Get the sum of electrons and positrons per slice, for every file in path. If path is a string, it is considered
    as a single file to analyze (ie a list with 1 entry).
    :param path: Path to file or list of paths from where to extract the particle counts.
    :type path: str or list
    :return np.ndarray: Particles counts, formatted as np.ndarray with dimensions (len(path), number_of_slices).
    """
    if type(path) == str:
        long = np.genfromtxt(path, skip_header=2, skip_footer=216, usecols=(2, 3))
        return np.sum(long, axis=1)
    elif type(path) == list:
        res = np.array([np.sum(np.genfromtxt(file, skip_header=2, skip_footer=216, usecols=(2, 3)), axis=1)
                        for file in path])
        return res


# Infer the number of slices and number of antennas present in the simulation
sim_dir = os.path.join(SIM_DIRECTORY, f'SIM{TEMPLATE_NR}_coreas/')
particles_slice = get_number_of_particles(os.path.join(SIM_DIRECTORY, f'DAT{TEMPLATE_NR}.long'))
n_slice = particles_slice.shape[-1]
n_antenna = int(len(os.listdir(sim_dir)) / n_slice)

# Get the Xmax of the simulation and construct the vector for matrix multiplication
with open(os.path.join(SIM_DIRECTORY, 'SIM' + TEMPLATE_NR + '.reas'), 'r') as file:
    for line in file:
        if 'DepthOfShowerMaximum' in line:
            x_max = float(line.split()[2])
        elif 'PrimaryParticleEnergy' in line:
            primaryE = float(line.split()[2])
x_max_vector = np.array([1, x_max, x_max**2])

# Find the frequency range in consideration, more specifically the number of frequencies and the number of polarizations
with open(os.path.join(sim_dir, f'raw_0x5.dat'), 'r') as file:
    data = np.genfromtxt(file) * c_vacuum * 1e2
freq = np.fft.rfftfreq(len(data), 2e-10)
f_range = np.logical_and(12 * 1e6 <= freq, freq <= 502 * 1e6)
n_freq = sum(f_range)
n_pol = 2

# Prepare all the Numpy arrays
A = np.zeros((n_antenna, n_slice, n_freq, n_pol))
Phi = np.zeros((n_antenna, n_slice, n_freq, n_pol))
A_temp = np.zeros((n_antenna, n_slice, n_freq, n_pol))
A0 = np.zeros((n_antenna, n_slice, n_pol))
b = np.zeros((n_antenna, n_slice, n_pol))
c = np.zeros((n_antenna, n_slice, n_pol))

# Analyze all the files of the template shower
os.chdir(sim_dir)
for slice_nr in range(n_slice):
    for antenna in range(n_antenna):
        with open(f'raw_{antenna}x{int((slice_nr + 1) * 5)}.dat', 'r') as file:
            data = np.genfromtxt(file) * c_vacuum * 1e2

        spectrum = np.apply_along_axis(np.fft.rfft, 0, data[:, 1:], norm='forward')
        amplitude = np.abs(spectrum) * 2  # Need to multiply by 2 because 1-sided FT
        phase = np.angle(spectrum)
        filtered = amplitude[f_range]

        A[antenna, slice_nr, :, :] = filtered[:, :2]
        Phi[antenna, slice_nr, :, :] = phase[f_range][:, :2]

    with open(os.path.join(PARAM_DIRECTORY, 'fitX', f'slice{int((slice_nr + 1) * 5)}.dat'), 'r') as fileX, \
            open(os.path.join(PARAM_DIRECTORY, 'fitY', f'slice{int((slice_nr + 1) * 5)}.dat'), 'r') as fileY:
        dataX = np.genfromtxt(fileX)
        dataY = np.genfromtxt(fileY)

        p_A0_x, p_A0_y = dataX[:, 1:4], dataY[:, 1:4]
        p_b_x, p_b_y = dataX[:, 4:7], dataY[:, 4:7]
        p_c_x, p_c_y = dataX[:, 7:], dataY[:, 7:]

        A0[:, slice_nr, 0], A0[:, slice_nr, 1] = np.matmul(p_A0_x, x_max_vector), np.matmul(p_A0_y, x_max_vector)
        b[:, slice_nr, 0], b[:, slice_nr, 1] = np.matmul(p_b_x, x_max_vector), np.matmul(p_b_y, x_max_vector)
        c[:, slice_nr, 0], c[:, slice_nr, 1] = np.matmul(p_c_x, x_max_vector), np.matmul(p_c_x, x_max_vector)

# Calculate normalization factors, with frequency in MHz
for i, f in enumerate(freq[f_range] / 1e6):
    A_temp[:, :, i, :] = A0 * np.exp(b * f) * np.exp(c * f**2)  # have to add d to this

A_temp[np.where(A_temp == 0)] = 1  # Avoid division by 0, due to overflow in exp
denominator = np.apply_along_axis(lambda ar: ar * particles_slice, 1, A_temp)

# Normalized amplitude spectrum
A_res = A / denominator
