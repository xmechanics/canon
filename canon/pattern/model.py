import numpy as np
import logging
from timeit import default_timer as timer
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from canon.pattern.labeler import SeqLabeler

_logger = logging.getLogger(__name__)


class Model:

    def __init__(self):
        self.__n_features = None
        self.__n_clusters = None
        self.__preprocessors = []
        self._estimator = None
        self._n_features_transformed = None
        self.__pca = PCA(n_components=3)
        self.__pca_range = None

    def centroids(self):
        raise NotImplementedError

    def delegates(self, features, scores=None):
        if scores is None:
            scores = np.array(self.score(features))
        X = features
        for preprocessor in self.__preprocessors:
            X = preprocessor.transform(X)
        delegates = []
        for cluster, centroid in enumerate(self.centroids()):
            idx = None
            gidx = np.where(scores == cluster)[0]
            if len(gidx) > 0:
                dists = np.linalg.norm(X[gidx] - centroid, axis=1)
                nearest = dists.argmin()
                idx = gidx[nearest]
            delegates.append(idx)
        return delegates

    def get_label_scaler(self, dim, scaling='samples'):
        centroids = self.centroids()[:]
        if scaling == 'samples':
            pca = self.__pca
            pca_range = self.__pca_range
        elif scaling == 'centroids':
            pca = PCA(n_components=3)
            centroids_pca = pca.fit_transform(centroids)
            pca_range = [centroids_pca.min(axis=0), centroids_pca.max(axis=0)]
        else:
            raise ValueError("Unknown scaling mode %s" % scaling)
        transformed_centroids = pca.transform(centroids)
        centroids_rgb = (transformed_centroids - pca_range[0]) / (pca_range[1] - pca_range[0])
        # centroids_rgb = normalize(centroids_rgb, axis=1)
        return np.vectorize(lambda z: (centroids_rgb[int(z), dim] if z >= 0 else [np.nan, np.nan, np.nan]))

    def color_by_pca(self, Z, scaling='samples'):
        rgb = np.zeros((Z.shape[0], Z.shape[1], 3))
        centroids = self.centroids()[:]
        if centroids.shape[0] == 2:
            sidx = centroids[:, 0].argsort()
            blue = np.zeros(Z.shape)
            blue[np.where(Z == sidx[0])] = 255
            rgb[:, :, 2] = blue
            red = np.zeros(Z.shape)
            red[np.where(Z == sidx[1])] = 255
            rgb[:, :, 0] = red
        else:
            if scaling == 'samples':
                transformed_centroids = self.__pca.transform(centroids)
                pca_range = self.__pca_range
            elif scaling == 'centroids':
                pca = PCA(n_components=3)
                transformed_centroids = pca.fit_transform(centroids)
                pca_range = [transformed_centroids.min(axis=0), transformed_centroids.max(axis=0)]
            else:
                raise ValueError("Unknown scaling mode %s" % scaling)
            centroids_rgb = (transformed_centroids - pca_range[0]) / (pca_range[1] - pca_range[0])
            rgb[np.where(Z >= 0)] = centroids_rgb[Z[np.where(Z >= 0)]]
        return rgb

    def score_by_seqs(self, features, seqfiles, scores=None):
        if scores is None:
            scores = np.array(self.score(features))
        X = features
        for preprocessor in self.__preprocessors:
            X = preprocessor.transform(X)
        labler = SeqLabeler(seqfiles)
        oridx = []
        for cluster, centroid in enumerate(self.centroids()):
            cval = None
            gidx = np.where(scores == cluster)[0]
            if len(gidx) > 0:
                dists = np.linalg.norm(X[gidx] - centroid, axis=1)
                near_to_far = dists.argsort()
                for lidx in near_to_far:
                    idx = gidx[lidx]
                    cval = labler.evaluate(idx)
                    if cval is not None and np.isfinite(cval):
                        break
            if cval is None or not np.isfinite(cval):
                _logger.warning("Cannot label cluster {} using SEQ".format(len(oridx)))
                print("Cannot label cluster {} using SEQ".format(len(oridx)))
            oridx.append(cval)
        oridx = np.array(oridx).astype('float32')
        img_shape = labler.img_shape()
        Z = oridx[scores].reshape(img_shape)
        return Z

    def compute_silhouette_score(self, data):
        X = data
        for preprocessor in self.__preprocessors:
            X = preprocessor.transform(X)
        return silhouette_score(data, self._estimator.predict(X))

    def compute_calinski_harabaz_score(self, data):
        X = data
        for preprocessor in self.__preprocessors:
            X = preprocessor.transform(X)
        return calinski_harabasz_score(data, self._estimator.predict(X))

    def train(self, data, preprocessors=None, n_clusters=None):
        n_patterns = len(data)
        n_features = len(data[0])
        self.__n_features = n_features

        t_start = timer()
        _logger.info('Pre-processing %d patterns with %d features ...' % (n_patterns, n_features))
        if preprocessors is None:
            preprocessors = []
        for preprocessor in preprocessors:
            data = preprocessor.fit_transform(data)
        self.__preprocessors = preprocessors

        data_pca = self.__pca.fit_transform(data)
        self.__pca_range = [data_pca.min(axis=0), data_pca.max(axis=0)]

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
        return np.array([record[0] for record in data])


class IterativeModel(Model):
    def __init__(self):
        Model.__init__(self)

    def _fit(self, samples, n_clusters=None):
        t_start = timer()

        iterative = False
        if n_clusters is None:
            n_clusters = min(len(samples), 256)
            iterative = True
        estimator = None
        terminate = False
        moving_forward = True  # forward means going towards 0 clusters
        max_clusters = n_clusters
        min_clusters = 0

        while not terminate:
            new_estimator = self._fit_step(samples, n_clusters)
            if estimator is None:
                estimator = new_estimator
                n_clusters = n_clusters // 2
                terminate = not iterative
            else:
                new_is_better = self._new_estimator_is_better(estimator, new_estimator, samples)
                n_clusters_2 = new_estimator.n_components
                if not new_is_better:
                    moving_forward = not moving_forward
                else:
                    estimator = new_estimator

                if moving_forward:
                    n_clusters = int((n_clusters_2 + min_clusters) / 2.)
                    max_clusters = n_clusters_2
                else:
                    n_clusters = int((n_clusters_2 + max_clusters) / 2.)
                    min_clusters = n_clusters_2

                if abs(n_clusters - n_clusters_2) < 4:
                    _logger.info("Terminate the iteration because n_clusters change is only {}".format(
                        n_clusters - n_clusters_2))
                    terminate = True

            if not terminate and (n_clusters > 256 or n_clusters < 16):
                reason = "few" if n_clusters < 16 else "many"
                _logger.info("Terminate the iteration because n_clusters {} is too {}".format(
                    n_clusters, reason))
                terminate = True

        n_clusters = estimator.n_components
        _logger.info('Finally got a model on %d patterns using %d features for %d clusters. %.3f sec' %
                     (len(samples), self._n_features_transformed, n_clusters, timer() - t_start))
        return estimator, n_clusters

    def _fit_step(self, samples, n_clusters):
        raise NotImplementedError("Need to implement _fit_step")

    # noinspection PyMethodMayBeStatic
    def _new_estimator_is_better(self, estimator, new_estimator, samples):
        return False


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
        estimator = KMeans(n_clusters=n_clusters, n_init=init, random_state=4)
        estimator.fit(samples)
        self._centroids = estimator.cluster_centers_
        _logger.info('Finished KMeans on %d samples using %d features for %d clusters. %.3f sec.' %
                     (len(samples), n_features, n_clusters, timer() - t_start))
        return estimator, n_clusters

    def _score_transformed_data(self, data):
        return self._estimator.predict(data)


class GMModel(IterativeModel):

    def __init__(self, min_prob=0.8):
        IterativeModel.__init__(self)
        self.__min_prob = min_prob

    def centroids(self):
        return self._estimator.means_

    def _fit_step(self, samples, n_clusters):
        t_start = timer()
        n_features = len(samples[0])
        _logger.info('Running GaussianMixture on %d patterns using %d features for %d clusters ...' %
                     (len(samples), n_features, n_clusters))
        estimator = GaussianMixture(n_components=n_clusters, random_state=4)
        estimator.fit(samples)
        _logger.info('Finished GaussianMixture on %d patterns using %d features for %d clusters. %.3f sec. AIC = %g' %
                     (len(samples), n_features, n_clusters, timer() - t_start,
                      estimator.aic(samples)))
        return estimator

    def _new_estimator_is_better(self, estimator, new_estimator, samples):
        aic = estimator.aic(samples)
        new_aic = new_estimator.aic(samples)
        return new_aic < aic

    def _score_transformed_data(self, data):
        labels = [-1] * len(data)
        probs = self._estimator.predict_proba(data)
        for i, p in enumerate(probs):
            max_p = np.max(p)
            if max_p >= self.__min_prob:
                labels[i] = np.argmax(p)
        return labels


class BGMModel(IterativeModel):
    def __init__(self, min_prob=0.8):
        IterativeModel.__init__(self)
        self.__min_prob = min_prob

    def centroids(self):
        return self._estimator.means_

    def _fit_step(self, samples, n_clusters):
        t_start = timer()
        n_features = len(samples[0])
        _logger.info('Running BayesianGaussianMixture on %d samples using %d features for %d clusters ...' %
                     (len(samples), n_features, n_clusters))
        estimator = BayesianGaussianMixture(n_components=n_clusters, covariance_type='full')
        estimator.fit(samples)
        _logger.info('Finished BayesianGaussianMixture on %d samples using %d features for %d clusters. %.3f sec. '
                     '90-percentile coverage = %g' %
                     (len(samples), n_features, n_clusters, timer() - t_start, self.ninety_percentile_coverage(estimator.weights_)))
        return estimator

    def _new_estimator_is_better(self, estimator, new_estimator, samples):
        coverage = self.ninety_percentile_coverage(estimator.weights_)
        new_coverage = self.ninety_percentile_coverage(new_estimator.weights_)
        return new_coverage > coverage

    @staticmethod
    def ninety_percentile_coverage(weights):
        ninety = np.percentile(weights, 90)
        above = weights[np.where(weights > ninety)]
        return float(np.sum(above) * len(weights) / len(above))

    def _score_transformed_data(self, data):
        labels = [-1] * len(data)
        probs = self._estimator.predict_proba(data)
        for i, p in enumerate(probs):
            max_p = np.max(p)
            if max_p >= self.__min_prob:
                # labels[i] = (np.argmax(p), max_p)
                labels[i] = np.argmax(p)
        return labels


class MeanShiftModel(Model):

    def __init__(self):
        Model.__init__(self)

    def centroids(self):
        return self._estimator.cluster_centers_

    def _fit(self, samples, n_clusters=None):
        t_start = timer()
        n_features = len(samples[0])
        _logger.info('Running MeanShift %d samples using %d features ...' % (len(samples), n_features))
        bandwidth = estimate_bandwidth(samples, quantile=0.2, n_samples=min(500, int(len(samples) * 0.1)))
        _logger.info("Estimated bandwidth is {}".format(bandwidth))
        estimator = MeanShift(bandwidth=bandwidth, cluster_all=False)
        estimator.fit(samples)
        n_clusters = len(estimator.cluster_centers_)
        _logger.info('Finished MeanShift on %d samples using %d features for %d clusters. %.3f sec.' %
                     (len(samples), n_features, n_clusters, timer() - t_start))
        return estimator, n_clusters

    def _score_transformed_data(self, data):
        return self._estimator.predict(data)


# class DBSCANModel(Model):
#
#     def __init__(self):
#         Model.__init__(self)
#
#     def centroids(self):
#         return self._estimator.cluster_centers_
#
#     def _fit(self, samples, n_clusters=None):
#         t_start = timer()
#         n_features = len(samples[0])
#         _logger.info('Running MeanShift %d samples using %d features ...' % (len(samples), n_features))
#         eps = estimate_bandwidth(samples, quantile=0.2, n_samples=min(500, int(len(samples) * 0.1)))
#         _logger.info("Estimated eps is {}".format(eps))
#         estimator = DBSCAN(eps=eps, min_samples=5)
#         estimator.fit(samples)
#         n_clusters = len(estimator.components_)
#         _logger.info('Finished MeanShift on %d samples using %d features for %d clusters. %.3f sec.' %
#                      (len(samples), n_features, n_clusters, timer() - t_start))
#         return estimator, n_clusters
#
#     def _score_transformed_data(self, data):
#         return self._estimator.fit_predict(data)
