def idx2XY(idx, NX):
    return int((idx -1) / NX), (idx - 1) % NX


def split_workload(loads, nproc):
    load_groups = [[] for _ in range(nproc)]
    for i, l in enumerate(loads):
        load_groups[i % len(load_groups)].append(l)
    return load_groups


def merge_workload(grps):
    merged = []
    tot_len = sum(map(len, grps))
    for i in range(tot_len):
        merged.append(grps[i % len(grps)][i / len(grps)])
    return merged
