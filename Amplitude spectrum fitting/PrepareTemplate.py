import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c as c_vacuum

SIM_DIRECTORY = '/mnt/hgfs/Shared data/BulkSynth/bulksynth-17/'
PARAM_DIRECTORY = '/home/mdesmet/PycharmProjects/cr-template-synthesis/Amplitude spectrum fitting/paramProfileFit50/'
TEMPLATE_NR = '100052'
TARGET_NR = '100013'


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
        res = np.array([np.sum(np.genfromtxt(particle_file, skip_header=2, skip_footer=216, usecols=(2, 3)), axis=1)
                        for particle_file in path])
        return res


# Infer the number of slices and number of antennas present in the template simulation
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
x_max_vector = np.array([1, x_max, x_max ** 2])

# Target shower parameters
particles_target = get_number_of_particles(os.path.join(SIM_DIRECTORY, f'DAT{TARGET_NR}.long'))
with open(os.path.join(SIM_DIRECTORY, 'SIM' + TARGET_NR + '.reas'), 'r') as file:
    for line in file:
        if 'DepthOfShowerMaximum' in line:
            x_max = float(line.split()[2])
x_max_vector_target = np.array([1, x_max, x_max ** 2])

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
A_synth = np.zeros((n_antenna, n_slice, n_freq, n_pol))

A0 = np.zeros((n_antenna, n_slice, n_pol))
b = np.zeros((n_antenna, n_slice, n_pol))
c = np.zeros((n_antenna, n_slice, n_pol))

A0_synth = np.zeros((n_antenna, n_slice, n_pol))
b_synth = np.zeros((n_antenna, n_slice, n_pol))
c_synth = np.zeros((n_antenna, n_slice, n_pol))

# Analyze all the files of the template shower
os.chdir(sim_dir)
for slice_nr in range(n_slice):
    for antenna_nr in range(n_antenna):
        with open(f'raw_{antenna_nr}x{int((slice_nr + 1) * 5)}.dat', 'r') as file:
            data = np.genfromtxt(file) * c_vacuum * 1e2

        spectrum = np.apply_along_axis(np.fft.rfft, 0, data[:, 1:])  # Normalisation must be consistent with IRFFT
        amplitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        A[antenna_nr, slice_nr, :, :] = amplitude[f_range][:, :n_pol]
        Phi[antenna_nr, slice_nr, :, :] = phase[f_range][:, :n_pol]

    with open(os.path.join(PARAM_DIRECTORY, 'fitX', f'slice{int((slice_nr + 1) * 5)}.dat'), 'r') as fileX, \
            open(os.path.join(PARAM_DIRECTORY, 'fitY', f'slice{int((slice_nr + 1) * 5)}.dat'), 'r') as fileY:
        dataX = np.genfromtxt(fileX)
        dataY = np.genfromtxt(fileY)

        p_A0_x, p_A0_y = dataX[:, 1:4], dataY[:, 1:4]
        p_b_x, p_b_y = dataX[:, 4:7], dataY[:, 4:7]
        p_c_x, p_c_y = dataX[:, 7:], dataY[:, 7:]

        A0[:, slice_nr, 0], A0[:, slice_nr, 1] = np.matmul(p_A0_x, x_max_vector), np.matmul(p_A0_y, x_max_vector)
        b[:, slice_nr, 0], b[:, slice_nr, 1] = np.matmul(p_b_x, x_max_vector), np.matmul(p_b_y, x_max_vector)
        c[:, slice_nr, 0], c[:, slice_nr, 1] = np.matmul(p_c_x, x_max_vector), np.matmul(p_c_y, x_max_vector)

        A0_synth[:, slice_nr, 0], A0_synth[:, slice_nr, 1] = \
            np.matmul(p_A0_x, x_max_vector_target), np.matmul(p_A0_y, x_max_vector_target)
        b_synth[:, slice_nr, 0], b_synth[:, slice_nr, 1] = \
            np.matmul(p_b_x, x_max_vector_target), np.matmul(p_b_y, x_max_vector_target)
        c_synth[:, slice_nr, 0], c_synth[:, slice_nr, 1] = \
            np.matmul(p_c_x, x_max_vector_target), np.matmul(p_c_y, x_max_vector_target)

# Calculate normalization factors, with frequency in MHz
for i, f in enumerate(freq[f_range] / 1e6):
    A_temp[:, :, i, :] = A0 * np.exp(b * f + c * f ** 2)  # have to add d to this
    A_synth[:, :, i, :] = A0_synth * np.exp(b_synth * f + c_synth * f ** 2)

A_temp[np.where(A_temp <= 1e-10)] = 1  # Avoid division by 0, due to overflow in exp
denominator = np.apply_along_axis(lambda ar: ar * particles_slice, 1, A_temp)

# Normalized amplitude spectrum
A_res = A / denominator
Phi_res = Phi

# Calculate the synthesized pulse
A_synth = np.apply_along_axis(lambda ar: ar * particles_target, 1, A_synth) * A_res
A_synth[np.isnan(A_synth)] = 0
E_synth = A_synth * np.exp(Phi_res * 1j)

# Compare pulses in filtered frequency band 12-502MHz
antenna = 2
# f_check_range = np.logical_and(30 * 1e6 <= freq[f_range], freq[f_range] <= 80 * 1e6)
# f_range = np.logical_and(30 * 1e6 <= freq, freq <= 80 * 1e6)

os.chdir(os.path.join(SIM_DIRECTORY, f'SIM{TARGET_NR}_coreas/'))
data = np.zeros([2082, 3])
for file in glob.glob(f'raw_{antenna}x*'):
    data += (np.genfromtxt(file) * c_vacuum * 1e2)[:, 1:]
spectrum = np.apply_along_axis(np.fft.rfft, 0, data)
filtered = np.apply_along_axis(lambda ar: ar * f_range, 0, spectrum)
signal = np.apply_along_axis(np.fft.irfft, 0, filtered)[:, :n_pol]

E_antenna = np.zeros((n_slice, spectrum.shape[0], n_pol)) + 1j * np.zeros((n_slice, spectrum.shape[0], n_pol))
E_antenna[:, f_range, :] = E_synth[antenna, :, :, :]
signal_synth = np.sum(np.apply_along_axis(np.fft.irfft, 1, E_antenna), axis=0)

time = np.genfromtxt(f'raw_{antenna}x5.dat', np.float32)[:, 0]

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

ax.plot(time * 1e9, np.real(signal[:, 0]), c='k', linestyle='--')
ax.plot(time * 1e9, np.real(signal_synth[:, 0]), c='maroon', linestyle='--')

ax.plot(time * 1e9, np.real(signal[:, 1]), label='Real', c='k')
ax.plot(time * 1e9, np.real(signal_synth[:, 1]), label='Synthesized', c='maroon')

ax.set_xlim([0, 10])
ax.set_title(r'$X_{max}^{Real}$ = ' + str(int(x_max_vector_target[1])) + r' $g/cm^2$ - '
             r'$X_{max}^{Temp}$ = ' + str(int(x_max_vector[1])) + r' $g/cm^2$ - '
             f'antenna = {antenna}')
ax.legend()

plt.show()
