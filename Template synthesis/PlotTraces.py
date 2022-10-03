import numpy as np
import matplotlib.pyplot as plt


ANTENNA = 3


times_0 = np.load('Traces/SIM100000_times.npy')
traces_0 = np.load('Traces/SIM100000_traces.npy')

times_1 = np.load('Traces/SIM100001_times.npy')
traces_1 = np.load('Traces/SIM100001_traces.npy')

fig1, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 6))

ax1.plot(times_0[ANTENNA], traces_0[ANTENNA, 0])
ax1.plot(times_1[ANTENNA], traces_1[ANTENNA, 0])

ax2.plot(times_0[ANTENNA], traces_0[ANTENNA, 1])
ax2.plot(times_1[ANTENNA], traces_1[ANTENNA, 1])

# plt.savefig("Traces/trace_0_1.png", bbox_inches='tight')
