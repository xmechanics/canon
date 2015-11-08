import numpy as np
import logging
from timeit import default_timer as timer
from sklearn.mixture import GMM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class Model:

    def __init__(self):
        self.__n_features = None
        self.__n_clusters = None
        self.__preprocessors = []
        self.__estimator = None
        self.__n_features_transformed = None

    def train(self, data, preprocessors=[StandardScaler(), PCA(whiten=True)]):
        n_patterns = len(data)
        n_features = len(data[0])
        self.__n_features = n_features

        t_start = timer()
        logging.debug('Preprocessing %d patterns with %d features ...' % (n_patterns, n_features))
        for preprocessor in preprocessors:
            data = preprocessor.fit_transform(data)
        self.__preprocessors = preprocessors

        n_features = len(data[0])
        self.__n_features_transformed = n_features
        logging.info('Finished preprocessing of %d patterns with %d features. %.3f sec' %
                     (n_patterns, n_features, timer() - t_start))

        self.gmm_fit_iter(data)

    def gmm_fit_iter(self, samples):
        t_start = timer()
        n_clusters = len(samples)
        best_estimator = None
        min_aic = None

        while n_clusters >= 16:
            n_clusters /= 2
            estimator = self.gmm_fit(samples, n_clusters)
            aic = estimator.aic(samples)
            if min_aic is None:
                min_aic = aic
            if aic > min_aic and min(abs(aic), abs(min_aic)) < 0.5 * max(abs(min_aic), abs(aic)):
                break
            elif aic <= min_aic:
                best_estimator, min_aic = estimator, aic

        self.__estimator = best_estimator
        self.__n_clusters = self.__estimator.n_components
        logging.info('Finally got a GMM model on %d patterns using %d features for %d clusters. %.3f sec. AIC = %g' %
                     (len(samples), self.__n_features_transformed, self.__n_clusters, timer() - t_start,
                      self.__estimator.aic(samples)))

    def gmm_fit(self, samples, n_clusters):
        t_start = timer()
        n_features = len(samples[0])
        logging.debug('Running GMM on %d patterns using %d features for %d clusters ...' %
                      (len(samples), n_features, n_clusters))
        estimator = GMM(n_components=n_clusters)
        estimator.fit(samples)
        logging.info('Finished GMM on %d patterns using %d features for %d clusters. %.3f sec. AIC = %g' %
                     (len(samples), n_features, n_clusters, timer() - t_start,
                      estimator.aic(samples)))
        return estimator

    def score(self, data, min_prob=0.1):
        if len(data[0]) != self.__n_features:
            raise ValueError('The number of features [%d] in the data is different from that in the model [%d].' %
                             (len(data[0]), self.__n_features))

        for preprocessor in self.__preprocessors:
            data = preprocessor.transform(data)

        if len(data[0]) != self.__n_features_transformed:
            raise ValueError('The number of transformed features [%d] in the data is different from that in the model [%d].' %
                             (len(data[0]), self.__n_features_transformed))

        labels = [None] * len(data)
        probs = self.__estimator.predict_proba(data)
        for i, p in enumerate(probs):
            max_p = np.max(p)
            if max_p >= min_prob:
                labels[i] = (np.where(p == max_p)[0][0], max_p)
        return labels




