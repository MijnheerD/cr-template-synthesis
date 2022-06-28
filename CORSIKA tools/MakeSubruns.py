import os
import shutil
import sys

from MakeInp import proton_sim, iron_sim, DIR, RUNS


def make_runs(nr_of_sims, directory=DIR, sim_start=0):
    # Change dir to find templates
    os.chdir(directory)

    for ind in range(sim_start, sim_start + nr_of_sims):
        # Generate the run numbers for the simulations
        proton_sim_nr = f'145{ind:03d}'
        iron_sim_nr = f'245{ind:03d}'

        # Create the INP files and place them in the subdirectories
        proton_sim(proton_sim_nr)
        iron_sim(iron_sim_nr)

        # Add LIST and REAS files to the subdirectories
        for run in RUNS:
            shutil.copyfile(f'{run}.list', os.path.join(run, f'SIM{proton_sim_nr}.list'))
            shutil.copyfile(f'SIMxxxxxx.reas', os.path.join(run, f'SIM{proton_sim_nr}.reas'))

            shutil.copyfile(f'{run}.list', os.path.join(run, f'SIM{iron_sim_nr}.list'))
            shutil.copyfile(f'SIMxxxxxx.reas', os.path.join(run, f'SIM{iron_sim_nr}.reas'))


if __name__ == "__main__":
    if len(sys.argv)>1:
        nr = sys.argv[1]
    else:
        nr = 3

    make_runs(3)
