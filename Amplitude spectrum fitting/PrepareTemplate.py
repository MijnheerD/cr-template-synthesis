import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c as c_vacuum

SIM_DIRECTORY = '/mnt/hgfs/Shared data/BulkSynth/bulksynth-17/'
PARAM_DIRECTORY = '/home/mdesmet/PycharmProjects/cr-template-synthesis/Parameters/paramProfileFit_LOFAR/'
F0 = 50
TEMPLATE_NR = '100001'
TARGET_NR = '100000'


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
template_dir = os.path.join(SIM_DIRECTORY, f'SIM{TEMPLATE_NR}_coreas/')
particles_temp = get_number_of_particles(os.path.join(SIM_DIRECTORY, f'DAT{TEMPLATE_NR}.long'))
n_slice = particles_temp.shape[-1]
n_antenna = int(len(os.listdir(template_dir)) / n_slice)

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
with open(os.path.join(template_dir, f'raw_0x5.dat'), 'r') as file:
    data = np.genfromtxt(file) * c_vacuum * 1e2
freq = np.fft.rfftfreq(len(data), 2e-10)
f_range = np.logical_and(30 * 1e6 <= freq, freq <= 80 * 1e6)
n_freq = sum(f_range)
n_time = len(data)
n_pol = 2

# Prepare all the Numpy arrays
A = np.zeros((n_antenna, n_slice, n_freq, n_pol))
Phi = np.zeros((n_antenna, n_slice, n_freq, n_pol))
A_temp = np.zeros((n_antenna, n_slice, n_freq, n_pol))
A_synth = np.zeros((n_antenna, n_slice, n_freq, n_pol))

A0 = np.zeros((n_antenna, n_slice, n_pol))
b = np.zeros((n_antenna, n_slice, n_pol))
c = np.zeros((n_antenna, n_slice, n_pol))

A0_target = np.zeros((n_antenna, n_slice, n_pol))
b_target = np.zeros((n_antenna, n_slice, n_pol))
c_target = np.zeros((n_antenna, n_slice, n_pol))

E_temp = np.zeros((n_antenna, n_slice, n_time, n_pol))

# Analyze all the files of the template shower
os.chdir(template_dir)
for slice_nr in range(n_slice):
    for antenna_nr in range(n_antenna):
        with open(f'raw_{antenna_nr}x{int((slice_nr + 1) * 5)}.dat', 'r') as file:
            data = np.genfromtxt(file) * c_vacuum * 1e2

        spectrum = np.apply_along_axis(np.fft.rfft, 0, data[:, 1:])  # Normalisation must be consistent with IRFFT
        amplitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        A[antenna_nr, slice_nr, :, :] = amplitude[f_range][:, :n_pol]
        Phi[antenna_nr, slice_nr, :, :] = phase[f_range][:, :n_pol]

        E_temp[antenna_nr, slice_nr, :, :] = np.apply_along_axis(np.fft.irfft, 0,
                                                                 spectrum * f_range[:, np.newaxis])[:, :n_pol]

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

        A0_target[:, slice_nr, 0], A0_target[:, slice_nr, 1] = \
            np.matmul(p_A0_x, x_max_vector_target), np.matmul(p_A0_y, x_max_vector_target)
        b_target[:, slice_nr, 0], b_target[:, slice_nr, 1] = \
            np.matmul(p_b_x, x_max_vector_target), np.matmul(p_b_y, x_max_vector_target)
        c_target[:, slice_nr, 0], c_target[:, slice_nr, 1] = \
            np.matmul(p_c_x, x_max_vector_target), np.matmul(p_c_y, x_max_vector_target)

# Bring A0 back to their unnormalized value
A0 = np.apply_along_axis(lambda ar: ar * particles_temp, 1, A0)
A0_target = np.apply_along_axis(lambda ar: ar * particles_target, 1, A0_target)

# Calculate the array containing the d values
distances = np.array([1, 4000, 7500, 11000, 15000, 37500])
slices = np.array([(ind + 1) * 5 for ind in range(n_slice)])

inner = 1e-9 * (slices[np.newaxis, :] / 400 - 1.5) * np.exp(1 - distances[:, np.newaxis] / 40000)
d = np.maximum(inner, np.zeros(inner.shape)) ** 2

# Calculate normalization factors, with frequency in MHz
for i, f in enumerate(freq[f_range] / 1e6):
    A_temp[:, :, i, :] = A0 * np.exp(b * (f - F0) + c * (f - F0) ** 2) + d[:, :, np.newaxis]
    A_synth[:, :, i, :] = A0_target * np.exp(b_target * (f - F0) + c_target * (f - F0) ** 2)

# Normalized amplitude spectrum
A_res = A / A_temp
Phi_res = Phi

# Calculate the synthesized pulse
# G_synth = np.apply_along_axis(lambda ar: ar * particles_target / particles_temp, 1, A_synth * A_res)
G_synth = A_synth * A_res
E_synth = G_synth * np.exp(Phi_res * 1j)

# Compare pulses in filtered frequency band 0-502MHz
os.chdir(os.path.join(SIM_DIRECTORY, f'SIM{TARGET_NR}_coreas/'))

fig, ax = plt.subplots(2, 2, figsize=(18, 14))
ax = ax.reshape((4,))
ax_x_lim = [(-5, 15), (0, 30), (0, 80)]
ax_y_lim = [(-250, 650), (-100, 300), (-2, 2)]

# Plot longitudinal profile
ax[0].plot(slices, particles_temp, c='purple', label='Template')
ax[0].plot(slices, particles_target, c='k', label='Real')
ax[0].set_xlabel(r"$X [g/cm^2]$")
ax[0].set_ylabel(r"$N(X)$")
ax[0].set_title(r'$X_{max}^{Real}$ = ' + str(int(x_max_vector_target[1])) + r' $g/cm^2$ - '
                r'$X_{max}^{Temp}$ = ' + str(int(x_max_vector[1])) + r' $g/cm^2$')
ax[0].legend()

for ind, antenna in enumerate((1, 3, 5)):
    data = np.zeros([n_time, 3])
    for file in glob.glob(f'raw_{antenna}*'):
        data += (np.genfromtxt(file) * c_vacuum * 1e2)[:, 1:]
    spectrum = np.apply_along_axis(np.fft.rfft, 0, data)
    filtered = np.apply_along_axis(lambda ar: ar * f_range, 0, spectrum)
    signal = np.apply_along_axis(np.fft.irfft, 0, filtered)[:, :n_pol]

    E_antenna = np.zeros((n_slice, spectrum.shape[0], n_pol)) + 1j * np.zeros((n_slice, spectrum.shape[0], n_pol))
    E_antenna[:, f_range, :] = E_synth[antenna, :, :, :]
    signal_synth = np.sum(np.apply_along_axis(np.fft.irfft, 1, E_antenna), axis=0)

    signal_temp = np.sum(E_temp, axis=1)[antenna, :, :]

    temp_time = np.genfromtxt(os.path.join('..', f'SIM{TEMPLATE_NR}_coreas/', f'raw_{antenna}x5.dat'),
                              np.float32)[:, 0] * 1e9
    target_time = np.genfromtxt(f'raw_{antenna}x5.dat', np.float32)[:, 0] * 1e9

    ax[ind+1].plot(target_time, np.real(signal[:, 0]), c='k', linestyle='--')
    ax[ind+1].plot(temp_time, np.real(signal_synth[:, 0]), c='maroon', linestyle='--')
    ax[ind+1].plot(temp_time, np.real(signal_temp[:, 0]), c='purple', linestyle='--')

    ax[ind+1].plot(target_time, np.real(signal[:, 1]), label='Real', c='k')
    ax[ind+1].plot(temp_time, np.real(signal_synth[:, 1]), label='Synthesized', c='maroon')
    ax[ind+1].plot(temp_time, np.real(signal_temp[:, 1]), label='Template', c='purple')

    ax[ind+1].set_xlim(ax_x_lim[ind])
    # ax[ind+1].set_ylim(ax_y_lim[ind])
    ax[ind+1].set_title(r'$X_{max}^{Real}$ = ' + str(int(x_max_vector_target[1])) + r' $g/cm^2$ - '
                        r'$X_{max}^{Temp}$ = ' + str(int(x_max_vector[1])) + r' $g/cm^2$ - '
                        f'r = {int(distances[antenna] / 100)} m')
    ax[ind+1].set_xlabel(r"t [ns]")
    ax[ind+1].set_ylabel(r"E [$\mu$ V/m]")
    ax[ind+1].legend()

plt.show()
'''
# Compare signals in freq space
antenna = 3
data = np.zeros([n_time, 3])
for file in glob.glob(f'raw_{antenna}*'):
    data += (np.genfromtxt(file) * c_vacuum * 1e2)[:, 1:]
spectrum = np.apply_along_axis(np.fft.rfft, 0, data)
filtered = np.apply_along_axis(lambda ar: ar * f_range, 0, spectrum)
signal = filtered[:, :n_pol]

E_antenna = np.zeros((n_slice, spectrum.shape[0], n_pol)) + 1j * np.zeros((n_slice, spectrum.shape[0], n_pol))
E_antenna[:, f_range, :] = E_synth[antenna, :, :, :]
signal_synth = np.sum(E_antenna, axis=0)

signal_scale = E_scale[antenna, :, :]

time = np.genfromtxt(f'raw_{antenna}x5.dat', np.float32)[:, 0]
x_axis = freq / 1e6

ax[1].plot(x_axis, np.abs(signal[:, 0]), label='Amplitude real', c='k', linestyle='--')
ax[1].plot(x_axis, np.abs(signal_synth[:, 0]), label='Amplitude synthesized', c='maroon', linestyle='--')

x = np.abs(signal_synth[:210, 0])/np.abs(signal[:210, 0])
print("Charge excess")
print(np.mean(x), np.std(x))

ax[2].plot(x_axis, np.abs(signal[:, 1]), label='Amplitude real', c='k')
ax[2].plot(x_axis, np.abs(signal_synth[:, 1]), label='Amplitude synthesized', c='maroon')

y = np.abs(signal_synth[:210, 1])/np.abs(signal[:210, 1])
print("Geomagnetic")
print(np.mean(y), np.std(y))

ax[3].plot(x_axis, np.angle(signal[:, 0]), c='grey', linestyle='--')
ax[3].plot(x_axis, np.angle(signal_synth[:, 0]), c='red', linestyle='--')
ax[3].plot(x_axis, np.angle(signal[:, 1]), label='Phase real', c='grey')
ax[3].plot(x_axis, np.angle(signal_synth[:, 1]), label='Phase synthesized', c='red')
# ax[3].plot(x_axis, Phi_res[antenna], label='Phase template', c='purple', linestyle=':')

ax[1].set_xlim([0, 500])
ax[1].set_yscale('log')
ax[1].legend()

ax[2].set_xlim([0, 500])
ax[2].set_yscale('log')
ax[2].set_xlabel(r"f [MHz]")
ax[2].set_ylabel(r"A [$\mu$ V/m]")
ax[2].legend()

ax[3].set_xlim([0, 500])
ax[3].set_xlabel(r"f [MHz]")
ax[3].set_ylabel(r"A [$\mu$ V/m]")
ax[3].legend()

fig.suptitle(r'$X_{max}^{Real}$ = ' + str(int(x_max_vector_target[1])) + r' $g/cm^2$ - '
             r'$X_{max}^{Temp}$ = ' + str(int(x_max_vector[1])) + r' $g/cm^2$ - '
             f'r = {int(distances[antenna] / 100)} m')

plt.show()


# Animate contributions to signal

for i in range(n_slice):
    for ind, antenna in enumerate((1, 3, 5)):
        data = np.zeros([n_time, 3])
        for file in glob.glob(f'raw_{antenna}*'):
            data += (np.genfromtxt(file) * c_vacuum * 1e2)[:, 1:]
        spectrum = np.apply_along_axis(np.fft.rfft, 0, data)
        filtered = np.apply_along_axis(lambda ar: ar * f_range, 0, spectrum)
        signal = np.apply_along_axis(np.fft.irfft, 0, filtered)[:, :n_pol]

        E_antenna = np.zeros((n_slice, spectrum.shape[0], n_pol)) + 1j * np.zeros((n_slice, spectrum.shape[0], n_pol))
        E_antenna[:, f_range, :] = E_synth[antenna, :, :, :]
        signal_synth = np.sum(np.apply_along_axis(np.fft.irfft, 1, E_antenna)[:(i+1)], axis=0)

        signal_scale = E_scale[antenna, :, :]

        time = np.genfromtxt(f'raw_{antenna}x5.dat', np.float32)[:, 0]

        ax[ind+1].clear()

        ax[ind+1].plot(time * 1e9, np.real(signal[:, 0]), c='k', linestyle='--')
        ax[ind+1].plot(time * 1e9, np.real(signal_synth[:, 0]), c='maroon', linestyle='--')
        ax[ind+1].plot(time * 1e9, np.real(signal_scale[:, 0]), c='green', linestyle='--')

        ax[ind+1].plot(time * 1e9, np.real(signal[:, 1]), label='Real', c='k')
        ax[ind+1].plot(time * 1e9, np.real(signal_synth[:, 1]), label='Synthesized', c='maroon')
        ax[ind+1].plot(time * 1e9, np.real(signal_scale[:, 1]), label='Scaled model', c='green')

        ax[ind+1].set_xlim(ax_x_lim[ind])
        ax[ind+1].set_ylim(ax_y_lim[ind])
        ax[ind+1].set_title(r'$X_{max}^{Real}$ = ' + str(int(x_max_vector_target[1])) + r' $g/cm^2$ - '
                            r'$X_{max}^{Temp}$ = ' + str(int(x_max_vector[1])) + r' $g/cm^2$ - '
                            f'r = {int(distances[antenna] / 100)} m')
        ax[ind+1].set_xlabel(r"t [ns]")
        ax[ind+1].set_ylabel(r"E [$\mu$ V/m]")
        ax[ind+1].legend()

    plt.savefig(f'/home/mdesmet/PycharmProjects/cr-template-synthesis/Amplitude spectrum fitting/animationFrames/frame{i}.png',
                bbox_inches='tight')
'''