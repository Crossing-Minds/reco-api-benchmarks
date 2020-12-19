import numpy

from xminds.compat import logger

from .config import MIN_RATING, MAX_RATING
from .utils import partition_int


class BaseSyntheticModel:
    """
    The synthetic model stores the necessary information to compute the raw rating of any
    user-item pair and defines the logic of the computation.
    The stored information is the "synthetic truth" (`users_truth` and `items_truth`) behind
    the ratings. It can either be embeddings, clusterings, layers of clusterings, ...
    """

    def __init__(self, users_truth, items_truth):
        """
        :param (nu,d)-array users_truth: truth 
        :param (ni,d)-array items_truth:
        :param (nu,)-int-array? users_cluster:
        :param (ni,)-int-array? items_cluster:
        """
        self.ratings_scaler = None
        self.n_users = users_truth.shape[0]
        self.n_items = items_truth.shape[0]
        self.users_truth = users_truth
        self.items_truth = items_truth

    def get_ratings(self, users_idx, items_idx):
        """
        :param (n,)-int-array users_idx:
        :param (n,)-int-array items_idx:
        :returns: (n,)-float-array raw ratings
        """
        users = self.users_truth[users_idx]
        items = self.items_truth[items_idx]
        ratings = self._get_ratings(users, items)
        return ratings

    def _get_ratings(self, users_truth, items_truth):
        """
        :param (n,d)-array users_truth: 
        :param (n,d)-array items_truth:
        :returns: (n,)-float-array unscaled ratings
        """
        raise NotImplementedError()

    def compute_dense_matrix(self):
        """
        Only for small synthetic datasets.
        """
        shape = (self.n_users, self.n_items)
        size = shape[0]*shape[1]
        if size > 10**6:
            logger.warning('Computing the dense matrix of %d ratings. May be memory heavy.', size)
        users_idx, items_idx = numpy.indices(shape)
        return self.get_ratings(users_idx.ravel(), items_idx.ravel()).reshape(shape)


class LinearEmbeddingsModel(BaseSyntheticModel):

    def __init__(self, users_embedding, items_embedding, decrease_factor=1.):
        """
        :param (nu,d)-float-array users_embedding:
        :param (ni,d)-float-array items_embedding:
        :param float? decrease_factor:
        """
        assert decrease_factor <= 1
        decrease = decrease_factor**numpy.arange(users_embedding.shape[1])
        super().__init__(users_embedding * decrease, items_embedding * decrease)

    def _get_ratings(self, users_truth, items_truth):
        return (users_truth * items_truth).sum(axis=1)


class PureClustersModel(BaseSyntheticModel):

    def __init__(self, users_cluster, items_cluster):
        """
        :param (nu,)-int-array users_cluster:
        :param (ni,)-int-array items_cluster:
        """
        super().__init__(users_cluster.reshape(-1, 1), items_cluster.reshape(-1, 1))

    def _get_ratings(self, users_truth, items_truth):
        return (users_truth == items_truth).ravel().astype(float)


class ClustersLayersModel(BaseSyntheticModel):

    def __init__(self, users_clusters, items_clusters, decrease_factor=1.):
        """
        :param (nu,d)-int-array users_clusters:
        :param (ni,d)-int-array items_clusters:
        """
        super().__init__(users_clusters,  items_clusters)
        assert decrease_factor <= 1
        self.decrease_factor = decrease_factor

    def _get_ratings(self, users_truth, items_truth):
        depth = users_truth.shape[1]
        depth_mul = self.decrease_factor**numpy.arange(depth)
        eq = users_truth == items_truth
        return (eq * depth_mul).sum(axis=1).astype(float)


class ClustersProductModel(BaseSyntheticModel):

    def __init__(self, users_clusters,  items_clusters, clusters_dim):
        """
        :param (nu,d)-int-array users_clusters:
        :param (ni,d)-int-array items_clusters:
        :param (d,)-int-array: numbers of clusters for each clutering layer
        """
        super().__init__(users_clusters,  items_clusters)

    def _get_ratings(self, users_truth, items_truth):
        return (users_truth == items_truth).all(axis=1).astype(float)


class RandomModel(BaseSyntheticModel):

    def _get_ratings(self, users_truth, items_truth):
        return numpy.random.uniform(MIN_RATING, MAX_RATING, size=users_truth.size)


class BaseSyntheticTruthSampler:
    """
    [Interface class]
    Base class for samplers of the synthetic truth of the model (embeddings, clusters, ...).
    The `sample` method returns a subclass of BaseSyntheticModel.
    """

    def __init__(self, dimension):
        """
        :param int-or-tuple-of-int dimension: different meanings depending on the synthetic model
        """
        pass

    def sample(self, n_users, n_items):
        """
        :param int n_users:
        :param int n_items:
        :returns: BaseSyntheticModel
        """
        raise NotImplementedError()


class ClusteredEmbeddingsRatingsSampler(BaseSyntheticTruthSampler):
    """
    Ratings are the dot product of user and item embeddings.

    Sampled embeddings are the sum of a cluster contribution and an individual contribution.
    Sampled cluster contributions are the sum of an orthogonal part (different clusters are
        orthogonal) and of a random part.
    """

    def __init__(self, d, n_clusters=None, ortho_fraction=1., cluster_scale=4., normalize=True):
        """
        :param int d: embeddings dimension
        :param int? n_clusters: must be <= `d`
        :param float? ortho_fraction: in [0, 1]. Controls how much cluster centroids are 
            forced to be orthogonal. 
            Note: the higher is the dimension, the more two random vectors are orthogonal.
                Thus, if `d` is very large, clusters will be almost orthongals even if
                `ortho_fraction` is 0
        :param float? cluster_scale: in [0, np.inf]. Controls cluster centroids importance in
            the embeddings.
            0 -> no clusters
            np.inf -> pure-clusters
        :param bool? normalize: wheter to normalize the embeddings (after summing the
            different contributions).
        """
        assert d > 0
        assert (n_clusters or d) <= d
        assert 0 <= ortho_fraction <= 1
        assert 0 <= cluster_scale
        self.d = d
        self.n_clusters = n_clusters or d
        self.ortho_fraction = ortho_fraction
        self.cluster_scale = cluster_scale
        self.normalize = normalize

    def sample(self, n_users, n_items):
        clusters_centroid = ((1 - self.ortho_fraction) *
                             self._sample_sphere_vectors(self.n_clusters, self.d))

        clusters_centroid += (self.ortho_fraction *
                              self._sample_orthonormal_vectors(self.n_clusters, self.d))

        users_cl_size = partition_int(n_users, self.n_clusters)
        items_cl_size = partition_int(n_items, self.n_clusters)
        if self.cluster_scale == numpy.inf:
            users_emb = numpy.zeros((n_users, self.d))
            items_emb = numpy.zeros((n_items, self.d))
        else:
            users_emb = self._sample_sphere_vectors(n_users, self.d)
            items_emb = self._sample_sphere_vectors(n_items, self.d)
            clusters_centroid *= self.cluster_scale
        users_emb += numpy.repeat(clusters_centroid, users_cl_size, axis=0)
        items_emb += numpy.repeat(clusters_centroid, items_cl_size, axis=0)

        if self.normalize:
            users_emb = self._normalize(users_emb)
            items_emb = self._normalize(items_emb)

        return self._build_ratings_factory(users_emb, items_emb)

    def _build_ratings_factory(self, users_emb, items_emb):
        return LinearEmbeddingsModel(users_emb, items_emb)

    @classmethod
    def _normalize(cls, v):
        return v / numpy.linalg.norm(v, axis=1).reshape(-1, 1)

    @classmethod
    def _sample_sphere_vectors(cls, n, d):
        x = numpy.random.randn(n, d)
        return cls._normalize(x)

    @classmethod
    def _sample_orthonormal_vectors(cls, n, d):
        return numpy.identity(d)[:n, :]


class EmbeddingsRatingsSampler(ClusteredEmbeddingsRatingsSampler):
    """ 
    Ratings are the dot product of user and item embeddings.
    Embeddings are sampled uniformly on the euclidean shpere.
    """

    def __init__(self, d):
        """
        :param int d: embeddings dimension
        """
        super().__init__(d, n_clusters=d, ortho_fraction=0, cluster_scale=0)


class DecreasingImportanceEmbeddingsRatingsSampler(EmbeddingsRatingsSampler):
    """ 
    Ratings of the pair (u,i) is r(u, i) = sum_{k=1}^d alpha^k x_uk y_ik
    Where x_u and y_i are embeddings sampled uniformly on the euclidean shpere 
    And alpha < 1
    """

    def __init__(self, d, decrease_factor=0.9):
        """
        :param int d: embeddings dimension
        :param float decrease_factor:
        """
        super().__init__(d)
        self.decrease_factor = decrease_factor

    def _build_ratings_factory(self, users_emb, items_emb):
        return LinearEmbeddingsModel(users_emb, items_emb, decrease_factor=self.decrease_factor)


class PureClustersRatingsSampler(BaseSyntheticTruthSampler):
    """
    Fully clustered ratings: 
    The rating of a pair user-item only depends on whether they share the same cluster or not.
    Clusters have the same size.
    """

    def __init__(self, n_clusters):
        """
        :param int n_clusters:
        """
        self.n_clusters = n_clusters

    def sample(self, n_users, n_items):
        cl = numpy.arange(self.n_clusters)
        users_cluster = numpy.repeat(cl, partition_int(n_users, self.n_clusters))
        items_cluster = numpy.repeat(cl, partition_int(n_items, self.n_clusters))
        return PureClustersModel(users_cluster, items_cluster)


class ClustersLayersRatingsSampler(BaseSyntheticTruthSampler):
    """
    Base class for "clustering layers" based samplers.
    Clustering layers based ratings are:
    r(u, i) = sum_{k=1}^d [x_uk == y_ik]
    Where x_u are the clusters of the user u at each level, and y_i is the same but for the item i
    """

    def __init__(self, dims, unbalanced_factor=None):
        if numpy.isscalar(dims):
            dims = (2,)*dims
        self.dims = numpy.array(dims)
        assert (self.dims >= 2).all()
        unbalanced_factor = unbalanced_factor or self.dims.size
        self.p = unbalanced_factor**numpy.arange(self.dims.max())

    def sample(self, n_users, n_items):
        users_clusters = self._sample_clusters(n_users)
        items_clusters = self._sample_clusters(n_items)
        return self._build_ratings_factory(users_clusters, items_clusters, self.dims)

    def _sample_clusters(self, n):
        clusters = numpy.empty((n, self.dims.size), dtype=int)
        for c, d in enumerate(self.dims):
            p = self.p[:d]
            clusters[:, c] = numpy.random.choice(d, size=n, p=p/p.sum())
        sorter = numpy.lexsort(list(clusters[:, :5].T)[::-1])  
        # sorting is only for easier analysis / visualization of the dataset
        return clusters[sorter, :]

    def _build_ratings_factory(self, users_clusters, items_clusters, dims):
        return ClustersLayersModel(users_clusters, items_clusters)


class ClustersProductRatingsSampler(ClustersLayersRatingsSampler):
    """
    The rating of a pair user-item only depends on whether they share the same cluster at
    each clustering layer.
    This is very close to pure clusters, only the ground-truth explanation change (multiple layers) 
    which has an impact when it comes to users/items features.
    Clusters are unbalanced so that even with a lot of layers, ratings aren't almost all zeros.
    """

    def _build_ratings_factory(self, users_clusters, items_clusters, dims):
        return ClustersProductModel(users_clusters, items_clusters, dims)


class DecreasingImportanceClustersLayersRatingsSampler(ClustersLayersRatingsSampler):
    """
    Similar to `DecreasingImportanceEmbeddingsRatingsSampler` but with clustering layers.
    r(u, i) = sum_{k=1}^d alpha^k [x_uk == y_ik]   (with alpah < 1)
    """

    def __init__(self, dims, decrease_factor=0.9, unbalanced_factor=1.):
        super().__init__(dims, unbalanced_factor=unbalanced_factor)
        self.decrease_factor = decrease_factor

    def _build_ratings_factory(self, users_clusters, items_clusters, dims):
        return ClustersLayersModel(
            users_clusters, items_clusters, decrease_factor=self.decrease_factor)


class RandomRatingsSampler(BaseSyntheticTruthSampler):
    """
    Random ratings sampler
    """

    def __init__(self):
        super().__init__(1)

    def sample(self, n_users, n_items):
        """
        :param int n_users:
        :param int n_items:

        :returns: BaseSyntheticModel
        """
        return RandomModel(numpy.empty((n_users, 1)), numpy.empty((n_items, 1)))
