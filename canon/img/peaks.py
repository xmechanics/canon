from skimage.feature import peak_local_max

__MAX_THRESHOLD = 0
__MIN_THRESHOLD = 3


def find_peaks(img, npeaks=float('inf')):
    threshold = lambda x: 10. ** (-x)
    log_th = __MIN_THRESHOLD
    peaks = peak_local_max(img, threshold_rel=threshold(log_th))

    while len(peaks) > npeaks and log_th > __MAX_THRESHOLD:
        log_th -= 0.2
        peaks = peak_local_max(img, threshold_rel=threshold(log_th))

    return peaks
