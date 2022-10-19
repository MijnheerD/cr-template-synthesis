import os
import sys
from StarShapedAntenna import projected_antenna_layout

DIR = "/home/mitjadesmet/Documents/CORSIKA input files/"
BASEFILE = os.path.join(DIR, "SIMxxxxxx.list")
MAX_ANTENNA = 5


def write_antenna(filename, antenna_coord):
    """
    Write a single antenna to the file with filename. The template from BASEFILE is used for the slicing. The antenna
    coordinates are used for the placement as well as the name of the antenna. The naming convention for the antennas is
    P{x-val}_N{y-val}x{slice}, where P and N are 0/1 depending on whether the x and y coordinate are negative/positive
    respectively. The x-val and y-val are the absolute values of the x and y coordinates.
    :param str filename: The name of the file to which to append the antenna
    :param tuple antenna_coord: The coordinates of antenna
    :return: None
    """
    with open(BASEFILE, 'r') as f:
        base_contents = f.readlines()

    antenna_name = f'{int(antenna_coord[0] >= 0)}{int(abs(antenna_coord[0]) / 100)}_' \
                   f'{int(antenna_coord[1] >= 0)}{int(abs(antenna_coord[1]) / 100)}'
    antenna_contents = [line.replace('7500.0 0.0', f'{antenna_coord[0]} {antenna_coord[1]}')
                            .replace('75x', f'{antenna_name}x')
                        for line in base_contents]

    with open(filename, "a") as f:
        f.writelines(antenna_contents)

        # End the antenna with a blank line
        f.write("\n")


def make_list(filename, antenna_x, antenna_y):
    """
    Write all the antennas to the file with filename. The antenna_x and antenna_y must contain the x and y coordinates
    of the antennas. The z coordinate is assumed to be always 0.
    :param filename: File write the antennas to. Must not exist already.
    :param antenna_x: The x coordinates of the antennas
    :param antenna_y: The y coordinates of the antennas
    :return: None
    """
    if os.path.isfile(f"{filename}.list"):
        print("File with this name already exists. Rename the file or choose another filename.")
    else:
        os.mknod(f"{filename}.list")

    for antenna_coord in zip(antenna_x, antenna_y):
        write_antenna(f"{filename}.list", antenna_coord)


def make_list_files(zenith, azimuth, arms):
    x_p, y_p, _ = projected_antenna_layout(azimuth, zenith, number_of_arms=arms, coreas=True)

    # Create a single array for antenna positions
    x_antenna = x_p.flatten()
    y_antenna = y_p.flatten()

    # Round out numerical errors
    x_antenna[abs(x_antenna) < 1e-10] = 0.0
    y_antenna[abs(y_antenna) < 1e-10] = 0.0

    # Split the arms into chunks of MAX_ANTENNA
    x_chunks = [x_antenna[i * MAX_ANTENNA:(i + 1) * MAX_ANTENNA]
                for i in range((len(x_antenna) + MAX_ANTENNA - 1) // MAX_ANTENNA)]
    y_chunks = [y_antenna[i * MAX_ANTENNA:(i + 1) * MAX_ANTENNA]
                for i in range((len(y_antenna) + MAX_ANTENNA - 1) // MAX_ANTENNA)]

    for ind, (x, y) in enumerate(zip(x_chunks, y_chunks)):
        make_list(f'RUN{ind+1}', x, y)


if __name__ == "__main__":
    if len(sys.argv) > 3:
        arm = int(sys.argv[3])
        zen = float(sys.argv[2])
    elif len(sys.argv) > 2:
        arm = 8
        zen = int(sys.argv[2])
    else:
        arm = 8
        zen = 45.0

    azi = 0.0

    # Change dir where to save LIST files
    os.chdir(DIR)
    make_list_files(zen, azi, arm)
