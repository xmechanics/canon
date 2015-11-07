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
        self.__scaler = None
        self.__pca = None
        self.__estimator = None
        self.__data = None

    def train(self, data):
        n_patterns = len(data)
        n_features = len(data[0])

        t_start = timer()
        logging.debug('Preprocessing %d patterns with %d features ...' % (n_patterns, n_features))

        scaler = StandardScaler()
        scaler.fit_transform(data)
        pca = PCA(n_components=None, whiten=True)
        pca.fit_transform(data)

        logging.info('Finished preprocessing of %d patterns with %d features. %.3f sec' %
                     (n_patterns, n_features, timer() - t_start))

        self.__data = data
        self.__n_features = n_features
        self.__scaler = scaler
        self.__pca = pca
        self.__n_clusters = None

        self.gmm_fit_iter()

        self.__data = []

    def gmm_fit_iter(self):
        t_start = timer()
        self.__n_clusters = len(self.__data)
        best_estimator = None
        max_aic = None

        while best_estimator is None or self.__n_clusters >= 16:
            self.__n_clusters /= 2
            estimator = self.gmm_fit()
            aic = estimator.aic(self.__data)
            if max_aic is None:
                max_aic = aic
            if abs(aic) < 0.5 * abs(max_aic):
                self.__n_clusters *= 2
                break
            else:
                if abs(aic) >= abs(max_aic):
                    best_estimator, max_aic = estimator, aic
        self.__estimator = best_estimator
        logging.info('Finally got a GMM model on %d patterns using %d features for %d clusters. %.3f sec. AIC = %g' %
                     (len(self.__data), self.__n_features, self.__n_clusters, timer() - t_start,
                      self.__estimator.aic(self.__data)))

    def gmm_fit(self):
        t_start = timer()
        logging.debug('Running GMM on %d patterns using %d features for %d clusters ...' %
                      (len(self.__data), self.__n_features, self.__n_clusters))
        estimator = GMM(n_components=self.__n_clusters)
        estimator.fit(self.__data)
        logging.info('Finished GMM on %d patterns using %d features for %d clusters. %.3f sec. AIC = %g' %
                     (len(self.__data), self.__n_features, self.__n_clusters, timer() - t_start,
                      estimator.aic(self.__data)))
        return estimator

    def score(self, data, min_prob=0.8):
        if len(data[0]) != self.__n_features:
            raise ValueError('The number of features [%d] in the data is different from that in the model [%d].' %
                             (len(data[0]), self.__n_features))

        self.__scaler.transform(data)
        self.__pca.transform(data)
        labels = [None] * len(data)
        probs = self.__estimator.predict_proba(data)
        for i, p in enumerate(probs):
            max_p = np.max(p)
            if max_p >= min_prob:
                labels[i] = np.where(p == max_p)[0][0]
        return labels




