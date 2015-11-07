import os.path


def idx2XY(idx, NX):
    return (int((idx -1) / NX), (idx - 1) % NX)


def split_workload(loads, nproc):
    load_groups = [[] for _ in xrange(nproc)]
    for i, l in enumerate(loads):
        load_groups[i % len(load_groups)].append(l)
    return load_groups


def plot_dat(file_path, NX, filenum = None):
    if filenum == None:
        index = []
        allpatterns = []
        for file in os.listdir(file_path):
            index.append(idx2XY(int(file[-9:-4]), NX))
            f = open(file_path+file, 'r')
            pattern = []
            for line in f:
                f.next()
                elems = line.split()
                x, y, z = (float(elems[0]), float(elems[1]), float(elems[3]))
                pattern.append([int(x), int(y), z])
            allpatterns.append(pattern)
            f.close()
    else:
        index = []
        allpatterns = []
        file_counter = 0
        for file in os.listdir(file_path):
            file_counter += 1
            if file_counter > filenum:
                break
            index.append(idx2XY(int(file[-9:-4]), NX))
            f = open(file_path+file, 'r')
            pattern = []
            f.next()
            for line in f:
                elems = line.split()
                x, y, z = (float(elems[0]), float(elems[1]), float(elems[3]))
                pattern.append([x, y, z])
            allpatterns.append(pattern)
            f.close()
    return allpatterns, index