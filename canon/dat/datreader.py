import os


def read_dats_indir(dir_path):
    filenames = []
    for file in os.listdir(dir_path):
        filenames.append(os.path.join(dir_path, file))
    return read_dats(filenames), filenames


def read_dats(files):
    patterns = []
    for file in files:
        f = open(file, 'r')
        pattern = []
        f.next()
        for line in f:
            elems = line.split()
            x, y, z = (int(float(elems[0])), int(float(elems[1])), float(elems[3]))
            pattern.append([x, y, z])
        patterns.append(pattern)
        f.close()
    return patterns


def idx2XY(idx, NX):
    return int((idx -1) / NX), (idx - 1) % NX

