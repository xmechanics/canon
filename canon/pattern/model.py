import numpy as np
import logging
from timeit import default_timer as timer
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.preprocessing import StandardScaler

_logger = logging.getLogger(__name__)


class Model:

    def __init__(self):
        self.__n_features = None
        self.__n_clusters = None
        self.__preprocessors = []
        self._estimator = None
        self._n_features_transformed = None

    def train(self, data, preprocessors=None, n_clusters=None):
        n_patterns = len(data)
        n_features = len(data[0])
        self.__n_features = n_features

        t_start = timer()
        _logger.info('Pre-processing %d patterns with %d features ...' % (n_patterns, n_features))
        if preprocessors is None:
            preprocessors = [StandardScaler()]
        for preprocessor in preprocessors:
            data = preprocessor.fit_transform(data)
        self.__preprocessors = preprocessors

        n_features = len(data[0])
        self._n_features_transformed = n_features
        _logger.info('Finished pre-processing of %d patterns with %d features. %.3f sec' %
                     (n_patterns, n_features, timer() - t_start))

        self._estimator, self.__n_clusters = self._fit(data, n_clusters=n_clusters)

    def _fit(self, data, n_clusters=0):
        return None, n_clusters

    def score(self, data):
        if len(data[0]) != self.__n_features:
            raise ValueError('The number of features [%d] in the data is different from that in the model [%d].' %
                             (len(data[0]), self.__n_features))

        for preprocessor in self.__preprocessors:
            data = preprocessor.transform(data)

        if len(data[0]) != self._n_features_transformed:
            raise ValueError(
                'The number of transformed features [%d] in the data is different from that in the model [%d].' %
                (len(data[0]), self._n_features_transformed))

        return self._score_transformed_data(data)

    def _score_transformed_data(self, data):
        return [record[0] for record in data]


class KMeansModel(Model):

    def __init__(self):
        Model.__init__(self)
        self._centroids = None
        # self._inertia = None

    def centroids(self):
        return self._centroids

    def _fit(self, samples, n_clusters=2, init=4):
        t_start = timer()
        self.__n_features = len(samples[0])
        n_features = self.__n_features
        _logger.info('Running KMeans on %d samples using %d features for %d clusters ...' %
                      (len(samples), n_features, n_clusters))
        estimator = KMeans(n_clusters=n_clusters, n_init=init)
        estimator.fit(samples)
        # estimator.fit_transform(samples)
        # estimator.fit_predict(samples)
        self._centroids = estimator.cluster_centers_
        # self._inertia = estimator.inertia_
        _logger.info('Finished KMeans on %d samples using %d features for %d clusters. %.3f sec.' %
                     (len(samples), n_features, n_clusters, timer() - t_start))
        return estimator, n_clusters

    def _score_transformed_data(self, data):
        return self._estimator.predict(data)


class MixtureModel(Model):

    def __init__(self, min_prob=0.8):
        Model.__init__(self)
        self.__min_prob = min_prob

    def _fit(self, samples, n_clusters=None):
        t_start = timer()
        if n_clusters is None:
            n_clusters = min(len(samples), 256)
        best_estimator = None
        min_aic = None

        while best_estimator is None or n_clusters >= 16:
            if best_estimator is not None:
                n_clusters = n_clusters // 2
            estimator = self.mixture_fit(samples, n_clusters)
            aic = estimator.aic(samples)
            if min_aic is None:
                min_aic = aic
            if aic > min_aic and min(abs(aic), abs(min_aic)) < 0.5 * max(abs(min_aic), abs(aic)):
                break
            elif aic <= min_aic:
                best_estimator, min_aic = estimator, aic

        n_clusters = best_estimator.n_components
        _logger.info('Finally got a GM model on %d patterns using %d features for %d clusters. %.3f sec. AIC = %g' %
                     (len(samples), self._n_features_transformed, n_clusters, timer() - t_start,
                      best_estimator.aic(samples)))
        return best_estimator, n_clusters

    def mixture_fit(self, samples, n_clusters):
        raise NotImplementedError("Need to implement mixture_fit")

    def _score_transformed_data(self, data):
        labels = [None] * len(data)
        probs = self._estimator.predict_proba(data)
        for i, p in enumerate(probs):
            max_p = np.max(p)
            if max_p >= self.__min_prob:
                labels[i] = (np.argmax(p), max_p)
        return labels


class GMModel(MixtureModel):

    def __init__(self, min_prob=0.8):
        MixtureModel.__init__(self, min_prob)

    def mixture_fit(self, samples, n_clusters):
        t_start = timer()
        n_features = len(samples[0])
        _logger.info('Running GaussianMixture on %d patterns using %d features for %d clusters ...' %
                      (len(samples), n_features, n_clusters))
        estimator = GaussianMixture(n_components=n_clusters)
        estimator.fit(samples)
        _logger.info('Finished GaussianMixture on %d patterns using %d features for %d clusters. %.3f sec. AIC = %g' %
                     (len(samples), n_features, n_clusters, timer() - t_start,
                      estimator.aic(samples)))
        return estimator


class BGMModel(MixtureModel):
    def __init__(self, min_prob=0.8):
        MixtureModel.__init__(self, min_prob)

    # def _fit(self, samples, n_clusters=None):
    #     t_start = timer()
    #     self._n_features = len(samples[0])
    #     adaptive_n_clusters = False
    #     if n_clusters is None:
    #         adaptive_n_clusters = True
    #         n_clusters = min(len(samples), 256)
    #     estimator = self.bgmm_fit(samples, n_clusters)
    #     if adaptive_n_clusters:
    #         cover_clusters = self.coveraging_clusters(estimator.weights_)
    #         while estimator.n_components > 16 and cover_clusters < 0.8 * estimator.n_components:
    #             new_estimator = self.bgmm_fit(samples, int(0.8 * estimator.n_components))
    #             if np.max(new_estimator.weights_) > 0.7:
    #                 break
    #             estimator = new_estimator
    #             cover_clusters = self.coveraging_clusters(estimator.weights_)
    #     n_clusters = estimator.n_components
    #     _logger.info('Finally got a BGM model on %d samples using %d features for %d clusters. %.3f sec' %
    #                  (len(samples), self._n_features_transformed, n_clusters, timer() - t_start))
    #     return estimator, n_clusters

    def mixture_fit(self, samples, n_clusters):
        t_start = timer()
        n_features = len(samples[0])
        _logger.info('Running BayesianGaussianMixture on %d samples using %d features for %d clusters ...' %
                      (len(samples), n_features, n_clusters))
        estimator = BayesianGaussianMixture(n_components=n_clusters, covariance_type='full')
        estimator.fit(samples)
        _logger.info('Finished GaussianMixture on %d samples using %d features for %d clusters. %.3f sec.'
                     % (len(samples), n_features, n_clusters, timer() - t_start))
        return estimator

    # def coveraging_clusters(self, weights):
    #     ws = np.flip(np.sort(weights), axis=0)
    #     w_cum = 0
    #     i = len(ws) - 1
    #     for i, w in enumerate(ws):
    #         w_cum += w
    #         if w_cum >= 0.95:
    #             break
    #     return i + 1


class MeanShiftModel(Model):

    def __init__(self):
        Model.__init__(self)

    def centroids(self):
        return self._estimator.cluster_centers_

    def _fit(self, samples, n_clusters=None):
        t_start = timer()
        n_features = len(samples[0])
        _logger.info('Running MeanShift %d samples using %d features ...' % (len(samples), n_features))
        estimator = MeanShift(cluster_all=False)
        estimator.fit(samples)
        n_clusters = len(estimator.cluster_centers_)
        _logger.info('Finished MeanShift on %d samples using %d features for %d clusters. %.3f sec.' %
                     (len(samples), n_features, n_clusters, timer() - t_start))
        return estimator, n_clusters

    def _score_transformed_data(self, data):
        return self._estimator.fit_predict(data)
