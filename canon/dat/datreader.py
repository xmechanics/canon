import os


class DatReader:
    def __init__(self):
        pass

    @staticmethod
    def read_dats(dir_path):
        filenames = []
        patterns = []
        for file in os.listdir(dir_path):
            filenames.append(file)
            f = open(dir_path+file, 'r')
            pattern = []
            for line in f:
                f.next()
                elems = line.split()
                x, y, z = (float(elems[0]), float(elems[1]), float(elems[3]))
                pattern.append([int(x), int(y), z])
            patterns.append(pattern)
            f.close()
        return patterns, filenames