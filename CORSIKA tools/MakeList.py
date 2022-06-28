import os
import sys
from StarShapedAntenna import R, antenna_layout, extract_arm

BASEFILE = "/home/mitjadesmet/Documents/CORSIKA input files/SIMxxxxxx.list"


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

    antenna_name = f'{int(antenna_coord[0] >= 0)}{int(abs(antenna_coord[0]))}_' \
                   f'{int(antenna_coord[1] >= 0)}{int(abs(antenna_coord[1]))}'
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


if __name__ == "__main__":
    if len(sys.argv) > 2:
        arm = int(sys.argv[2])
    else:
        arm = 0

    x_r, y_r = antenna_layout(R)
    x_arm, y_arm = extract_arm(x_r, y_r, arm=arm)

    x_antenna = x_arm.flatten()
    y_antenna = y_arm.flatten()

    # Round out numerical errors
    x_antenna[abs(x_antenna) < 1e-13] = 0.0
    y_antenna[abs(y_antenna) < 1e-13] = 0.0

    make_list(sys.argv[1], x_antenna, y_antenna)
