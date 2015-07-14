import os

__resource_dir__ = os.path.dirname(__file__) + "/resources/"


def resource(filename):
    return __resource_dir__ + filename
