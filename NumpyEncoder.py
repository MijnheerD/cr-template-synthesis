"""
Store data in json format in a file/encode
"""

import numpy as np
import json


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  # This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def write_file_json(location, data):
    """serialization of python data for easy storage and transmission"""
    dumped = json.dumps(data, cls=NumpyEncoder)

    with open(location, "w") as write_file:
        json.dump(dumped, write_file, indent=4)  # since json accept str input


def read_file_json(location):
    """deserialization of json data to python data. returns a python dictionary"""
    with open(location, "r") as read_file:
        json_string = json.load(read_file)
    json_string = json.loads(json_string)
    return json_string
