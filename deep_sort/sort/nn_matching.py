# vim: expandtab:ts=4:sw=4
import numpy as np


# calculate the euclidean distance
def _pdist(a, b):
    # to calculate paired square distance
    # a NxM represents N objects, each of which has M values for embedding comparison
    # b LxM Represents L objects, each of which has M values for embedding comparison
    # return N x L matrix，for example dist[i][j] represents 
    # sum of squares distance between a[i] and b[j]
    # look up here：https://blog.csdn.net/frankzd/article/details/80251042
    """Compute pair-wise squared distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    a, b = np.asarray(a), np.asarray(b)  # copy data
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    # obtain the squaired sum of each embedding 
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    # sum(N) + sum(L) -2 x [NxM]x[MxL] = [NxL]
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2

# calculate the cosine distance 
def _cosine_distance(a, b, data_is_normalized=False):
    # calc the cosine d between a and b
    # a : [NxM] b : [LxM]
    # consine distance = 1 - consine similarity
    # https://blog.csdn.net/u013749540/article/details/51813922
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    if not data_is_normalized:
        # It is necessary to transform cosine similarity 
        # into cosine distance similar to Euclidean distance
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        #  np.linalg.norm is used to find the norm of vectors，
        # L2 norm is the default setting，euqals to calculating the Euclidean distance。
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def _nn_euclidean_distance(x, y):
    # Nearest neighbor Euclidean distance
    """ Helper function for nearest neighbor distance metric (Euclidean).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest Euclidean distance to a sample in `x`.

    """
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))  # to find the minimum


def _nn_cosine_distance(x, y):
    # find the nearest neighbor distance
    """ Helper function for nearest neighbor distance metric (cosine).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest cosine distance to a sample in `x`.

    """
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)


class NearestNeighborDistanceMetric(object):
    # For each target, return the nearest distance
    """
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.

    Parameters
    ----------
    metric : str
        Either "euclidean" or "cosine".
    matching_threshold: float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.

    Attributes
    ----------
    samples : Dict[int -> List[ndarray]]
        A dictionary that maps from target identities to the list of samples
        that have been observed so far.

    """

    def __init__(self, metric, matching_threshold, budget=None):
        # default matching_threshold = 0.2 budge = 100
        if metric == "euclidean":
            # apply NNE
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            # apply NNC
            self._metric = _nn_cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")

        # matching_threshold is called in a cascade matching function
        self.matching_threshold = matching_threshold
        # budge control hold how many features simultaneously
        self.budget = budget
        # samples is a dict {id->feature list}
        self.samples = {}
        

    def partial_fit(self, features, targets, active_targets):
        # function：partial fit，update the measured distance with new data
        # call：Called in the feature set update module part，namely, tracker.update()
        """Update the distance metric with new data.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : ndarray
            An integer array of associated target identities.
        active_targets : List[int]
            A list of targets that are currently present in the scene.
        """
        for feature, target in zip(features, targets):
            # add new feature for corresponding object，update feature set
            self.samples.setdefault(target, []).append(feature)
            # object id:  feature list
            # Set the budget, the maximum number of goals per class, and ignore it directly
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]

        # filter active samples
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        # function：compare the distance between the features and targets，return a cost matrix
        # call：during matching process，package distance as gated_metric,
        #       combine appearance info(features obtained by reid)+
        #       motion info(Mahalanobis distance is used to measure the similarity of two distributions)
        """Compute distance between features and targets.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : List[int]
            A list of targets to match the given `features` against.

        Returns
        -------
        ndarray
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest distance between
            `targets[i]` and `features[j]` under certain metric.

        """
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix
