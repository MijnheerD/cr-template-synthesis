"""
Base class to provide read/write functionality in bunch, using internal directory structure.
"""

import os


class ReaderWriter(object):
    def __init__(self):
        self.filesX = []
        self.filesY = []

    def open_all_files(self, path):
        """
        Opens all files in a directory and stores their file handles.
        :param str path: Path to the directory
        """
        for file in os.listdir(os.path.join(path, 'fitX')):
            self.filesX.append(open(os.path.join(path, 'fitX', file)))
            self.filesY.append(open(os.path.join(path, 'fitY', file)))

    def close_all_files(self):
        """
        Close all the files opened by the instance.
        """
        for file in self.filesX:
            file.close()

        for file in self.filesY:
            file.close()
