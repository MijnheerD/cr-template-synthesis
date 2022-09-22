import os
import shutil
import sys

import numpy as np
import matplotlib.pyplot as plt

from MakeInp import proton_sim, iron_sim, DIR, RUNS


def make_runs(nr_of_sims, directory=DIR, sim_start=0):
    # Change dir to find templates
    os.chdir(directory)

    # Make sure all subdirectories exist
    for run in RUNS:
        os.makedirs(run, exist_ok=True)

    # Loop over all the simulations
    proton_energies = []
    iron_energies = []
    for ind in range(sim_start, sim_start + nr_of_sims):
        # Generate the run numbers for the simulations
        proton_sim_nr = f'145{ind:03d}'
        iron_sim_nr = f'245{ind:03d}'

        # Create the INP files and place them in the subdirectories
        proton_energies.append(float(proton_sim(proton_sim_nr)))
        iron_energies.append(float(iron_sim(iron_sim_nr)))

        # Add LIST and REAS files to the subdirectories
        for run in RUNS:
            shutil.copyfile(f'{run}.list', os.path.join(run, f'SIM{proton_sim_nr}.list'))
            shutil.copyfile(f'SIMxxxxxx.reas', os.path.join(run, f'SIM{proton_sim_nr}.reas'))

            shutil.copyfile(f'{run}.list', os.path.join(run, f'SIM{iron_sim_nr}.list'))
            shutil.copyfile(f'SIMxxxxxx.reas', os.path.join(run, f'SIM{iron_sim_nr}.reas'))

    return proton_energies, iron_energies


if __name__ == "__main__":
    if len(sys.argv) > 1:
        nr = int(sys.argv[1])
    else:
        nr = 3

    p, i = make_runs(nr)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    bins = np.power(10, np.linspace(8, 10, 50))
    ax[0].hist(p, bins=bins)
    ax[1].hist(p, bins=bins)

    ax[0].set_xlabel('Energy (GeV)')
    ax[0].set_ylabel('Counts')
    ax[0].set_xscale('log')
    ax[0].set_title('Proton E distribution')

    ax[1].set_xlabel('Energy (GeV)')
    ax[1].set_ylabel('Counts')
    ax[1].set_xscale('log')
    ax[1].set_title('Iron log E distribution')

    fig.savefig(f"Edist_{nr}.png", bbox_inches='tight')
