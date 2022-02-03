import os
import sys
from typing import TextIO


def write_antenna_line(file: TextIO, antenna, line):
    file.write(f"AntennaPosition = {(antenna + 1) * 10.0} 0.0 0.0 ")
    file.write(f"{antenna}x{int(line * 5 + 5)} ")
    file.write(f"slantdepth {(line) * 5.0} {(line + 1) * 5.0}")
    file.write("\n")


def write_antenna(filename, antenna_number):
    with open(filename, "a") as f:
        for line_number in range(208):
            write_antenna_line(f, antenna_number, line_number)

        # End the antenna with a blank line
        f.write("\n")


def make_list(filename, nr_of_antennas):
    if os.path.isfile(f"{filename}.list"):
        print("File with this name already exists. Rename the file or choose another filename.")
    else:
        os.mknod(f"{filename}.list")

    for antenna_nr in range(nr_of_antennas):
        write_antenna(f"{filename}.list", antenna_nr)


if __name__ == "__main__":
    make_list(sys.argv[1], int(sys.argv[2]))
