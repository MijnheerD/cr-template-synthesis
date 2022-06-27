import os
import sys
from LogEdistribution import loguniform

DIR = "/home/mitjadesmet/Documents/CORSIKA input files/"
RUNS = ["RUN1", "RUN2", "RUN3", "RUN4"]


def sub_sim(sim_nr, sub, primary='proton'):
    basefile = f'{sub}_{primary}.inp'
    with open(basefile, 'r') as f:
        base_contents = f.readlines()

    # Change RUNNR
    # Change SEED
    # Change ERANGE
    contents = base_contents

    return contents


def proton_sim(sim_number, sub_dir):
    file_contents = sub_sim(sim_number, sub_dir, primary='proton')

    with open(os.path.join(sub_dir, f'SIM{sim_number}.inp'), 'w') as file:
        file.writelines(file_contents)


def iron_sim(sim_number, sub_dir):
    file_contents = sub_sim(sim_number, sub_dir, primary='iron')

    with open(os.path.join(sub_dir, f'SIM{sim_number}.inp'), 'w') as file:
        file.writelines(file_contents)


if __name__ == "__main__":
    # Change dir to find templates
    os.chdir(DIR)

    # 300 sims per primary (changed to 3 for testing)
    for ind in range(3):
        proton_sim_nr = f'145{ind:03d}'
        iron_sim_nr = f'245{ind:03d}'

        for run_nr in RUNS:
            proton_sim(proton_sim_nr, run_nr)
            iron_sim(iron_sim_nr, run_nr)
