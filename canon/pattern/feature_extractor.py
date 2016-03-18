import numpy as np


class FeaturesExtractor:

    def __init__(self):
        self.__n_features = None

    def features(self, patterns):
        pass

    def n_features(self):
        return self.__n_features

    def _set_n_features(self, n):
        self.__n_features = n


class AllPeaksExtractor(FeaturesExtractor):
    def __init__(self, sample_patterns, intensity_threshold=0.5, gaussion_height=10, gaussian_width=30):
        FeaturesExtractor.__init__(self)
        self.__threshold = intensity_threshold
        self.__height = gaussion_height
        self.__width = gaussian_width
        self.__all_peaks = self.peak_union(sample_patterns)
        self._set_n_features(len(self.__all_peaks))

    def features(self, pattern):
        peaks = _peaks_above_threshold(pattern, self.__threshold)
        gaussians = [AllPeaksExtractor.__gaussian2d(p[0], p[1], 1.0 * p[2], self.__height, self.__width)
                 for p in peaks]

        def intensity(X, Y):
            sum = np.zeros_like(X, dtype='f')
            for g in gaussians:
                sum += g(X, Y)
            return sum

        intensities = intensity(self.__all_peaks[:, 0], self.__all_peaks[:, 1])
        return intensities.tolist()

    def peak_union(self, patterns):
        peaks_set = set()
        for pattern in patterns:
            spots = _peaks_above_threshold(pattern, self.__threshold)
            peaks = set([(p[0], p[1]) for p in spots])
            peaks_set = AllPeaksExtractor.__merge_peak_sets(peaks_set, peaks)
        return np.array(list(peaks_set))

    @staticmethod
    def __merge_peak_sets(set1, set2, r_nb=3):
        # make sure set2 is smaller
        if len(set1) < len(set2):
            set1, set2 = set2, set1

        for p2 in set2:
            not_in_set1 = True
            for n2 in AllPeaksExtractor.__neighbors(p2, r_nb):
                if n2 in set1:
                    not_in_set1 = False
                    break
            if not_in_set1:
                set1.add(p2)

        return set1

    @staticmethod
    def __neighbors(p, r=1):
        yield p
        for x in xrange(-r, r + 1):
            ymax = int(np.floor(np.sqrt(r**2 - x**2)))
            for y in xrange(-ymax, ymax + 1):
                yield (p[0] + x, p[1] + y)

    @staticmethod
    def __gaussian2d(x0, y0, i, height=1, width=5):
        sigma = width
        A = i * height

        def gx(X):
            X0 = np.ones(X.shape, dtype='f') * x0
            return (X -X0) * (X - X0) / (2 * sigma ** 2)

        def gy(Y):
            Y0 = np.ones(Y.shape, dtype='f') * y0
            return (Y -Y0) * (Y - Y0) / (2 * sigma ** 2)

        def g(X, Y):
            return A * np.exp(-(gx(X) + gy(Y)))

        return g


class PeaksNumberExtractor(FeaturesExtractor):
    def __init__(self, intensity_threshold=0.0):
        FeaturesExtractor.__init__(self)
        self.__threshold = intensity_threshold
        self._set_n_features(1)

    def features(self, pattern):
        peaks = _peaks_above_threshold(pattern, self.__threshold)
        return [float(len(peaks))]


class MaxPeaksExtractor(FeaturesExtractor):
    def __init__(self):
        FeaturesExtractor.__init__(self)

    def features(self, pattern):
        peaks = sorted(_peaks_above_threshold(pattern, 0.0), key=lambda p: p[2], reverse=True)
        max_peak = peaks[0]
        return [max_peak[0], max_peak[1]]


    def __gaussian2d(self, x0, y0, i, height=1, width=5):
        sigma = width
        A = i * height

        def gx(X):
            X0 = np.ones(X.shape, dtype='f') * x0
            return (X -X0) * (X - X0) / (2 * sigma ** 2)

        def gy(Y):
            Y0 = np.ones(Y.shape, dtype='f') * y0
            return (Y -Y0) * (Y - Y0) / (2 * sigma ** 2)

        def g(X, Y):
            return A * np.exp(-(gx(X) + gy(Y)))

        return g


class CombinedExtractor(FeaturesExtractor):

    def __init__(self, extractors):
        FeaturesExtractor.__init__(self)
        self.__extractors = extractors

    def features(self, pattern):
        fts = []
        for extractor in self.__extractors:
            fts += extractor.features(pattern)
        return fts


def _peaks_above_threshold(pattern, threshold):
    if len(pattern) > 0:
        imax = max([p[2] for p in pattern])
        ith = threshold * imax
        return [(p[0], p[1], p[2]/imax) for p in pattern if p[2] >= ith]
    else:
        return [[0.0, 0.0, 0.0]]





