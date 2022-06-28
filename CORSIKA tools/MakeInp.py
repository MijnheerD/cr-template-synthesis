import os
import sys
import numpy as np
from LogEdistribution import loguniform

DIR = "/home/mitjadesmet/Documents/CORSIKA input files/"
RUNS = ["RUN1", "RUN2", "RUN3", "RUN4"]


def sample_primary_energy():
    """
    Sample a single primary energy from log-uniform distribution and format it for use in CORSIKE INP file.
    :return: Formatted energy string
    """
    primary_energy_exp = loguniform(17, 19, size=1, return_exp=True)  # random exponent in eV
    primary_energy_exp -= 9  # set in GeV
    if primary_energy_exp > 9:
        # Subtract the highest power possible
        primary_energy_rest = primary_energy_exp - 9
        # Calculate the prefactor
        primary_energy_pre = 10 ** primary_energy_rest
        # Format the string with the exponent
        primary_energy = f'{primary_energy_pre[0]:.3f}E+09'
    else:
        primary_energy_rest = primary_energy_exp - 8
        primary_energy_pre = 10 ** primary_energy_rest
        primary_energy = f'{primary_energy_pre[0]:.3f}E+08'
    return primary_energy


def sub_sim(sim_nr, sim_energy, sim_seed, sub, primary='proton'):
    """
    Find the simulation parameters to the INP file in the subdirectory sub for a specific primary. The current
    working directory must contain the template files.
    :param sim_nr: To be used for the RUNNR parameter
    :param sim_energy: To be used for the ERANGE parameter
    :param sim_seed: To be used for the SEED parameters
    :param sub: Subdirectory for the specific run
    :param primary: Name of the primary, used to identify correct template file
    :return: The contents to be written to the INP file
    """
    basefile = f'{sub}_{primary}.inp'
    with open(basefile, 'r') as f:
        base_contents = f.readlines()
    contents = base_contents

    # Change RUNNR
    contents[0] = base_contents[0].replace('145999', str(sim_nr))
    # Change SEED with random numbers
    contents[3] = base_contents[3].replace('1', str(sim_seed[0]))
    contents[4] = base_contents[4].replace('2', str(sim_seed[1]))
    contents[5] = base_contents[5].replace('3', str(sim_seed[2]))
    # Change ERANGE
    contents[7] = base_contents[7].replace('1.000E+08 1.000E+08', f'{sim_energy} {sim_energy}')

    return contents


def proton_sim(sim_number):
    """
    Create the input files for a proton simulation with sim_number in the subdirectories listed in RUNS. The primary
    energy and 3 seeds are randomly generated using sample_primary_energy() and numpy.random.randint() respectively.
    :param sim_number: The unique number of the simulation run.
    :return: None
    """
    sim_primary_energy = sample_primary_energy()
    sim_seeds = np.random.randint(1, 900000001, size=3)

    for sub_dir in RUNS:
        file_contents = sub_sim(sim_number, sim_primary_energy, sim_seeds, sub_dir, primary='proton')

        with open(os.path.join(sub_dir, f'SIM{sim_number}.inp'), 'w') as file:
            file.writelines(file_contents)


def iron_sim(sim_number):
    """
    Create the input files for a iron simulation with sim_number in the subdirectories listed in RUNS. The primary
    energy and 3 seeds are randomly generated using sample_primary_energy() and numpy.random.randint() respectively.
    :param sim_number: The unique number of the simulation run.
    :return: None
    """
    sim_primary_energy = sample_primary_energy()
    sim_seeds = np.random.randint(1, 900000001, size=3)

    for sub_dir in RUNS:
        file_contents = sub_sim(sim_number, sim_primary_energy, sim_seeds, sub_dir, primary='iron')

        with open(os.path.join(sub_dir, f'SIM{sim_number}.inp'), 'w') as file:
            file.writelines(file_contents)


def make_inp(nr_of_sims, directory=DIR, sim_start=0):
    """
    Create the input files for an equal number of proton and iron simulations. The working directory is changed to be
    directory and should contain the templates for the INP files. The simulation numbers are generated starting from
    sim_start. Note that sim_start+nr_of_sims should be lower than 999 for consistent behaviour.
    :param sim_start: Lowest number to use for counting the simulations.
    :param nr_of_sims: Number of simulations to generate the input files for.
    :param directory: Working directory
    :return: None
    """
    # Change dir to find templates
    os.chdir(directory)

    for ind in range(sim_start, sim_start+nr_of_sims):
        proton_sim_nr = f'145{ind:03d}'
        iron_sim_nr = f'245{ind:03d}'

        proton_sim(proton_sim_nr)
        iron_sim(iron_sim_nr)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        nr = sys.argv[1]
    else:
        nr = 3

    make_inp(nr)
