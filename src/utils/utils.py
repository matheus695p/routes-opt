import os


def ls_directory(path="data/input-data/gps"):
    files = os.listdir(path)
    return files
