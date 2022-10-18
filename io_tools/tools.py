import os
from pathlib import Path
from contextlib import contextmanager


@contextmanager
def working_directory(path: Path):
    """Sets the cwd within the context

    Args:
        path (Path): The path to the cwd

    Yields:
        None
    """

    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)


class FileWrapper:
    def __init__(self, fpath, fmode):
        self.dir, self.file = os.path.split(fpath)
        self.mode = fmode
        self.contents = None

    # Implement the context manager protocol
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    # Implement read/write functions
    def read(self):
        pass

    def write(self):
        pass
