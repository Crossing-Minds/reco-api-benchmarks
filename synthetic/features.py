import numpy

from xminds.lib.arrays import to_structured


class BaseFeatureSampler:
    """
    [Interface class]
    Base class for feature samplers. 
    """
    NAME = 'feature'

    def __init__(self, difficulty=0):
        self.difficulty = difficulty

    def sample(self, ground_turth_dim):
        """
        :param (n,)-array ground_turth_dim: One dimension/layer of the ground truth
            (either users or items ground truth)
        :return array-or-sparse-matrix: a feature of shape (n,m) based on the given
            `ground_turth_dim`
        """
        raise NotImplementedError()


class TagsSampler(BaseFeatureSampler):
    """
    Samples a "clustered m2m" in function of the given clusters. 
    """
    NAME = 'tags'
    IS_M2M = True
    DTYPE = 'u4'

    def __init__(self, difficulty=0, clusters_tag_dim=None, n_tags_range=(2, 6),
                 unbalanced_factor=1):
        super().__init__(difficulty)
        self.clusters_tag_dim = clusters_tag_dim
        self.n_tags_range = n_tags_range
        self.unbalanced_factor = unbalanced_factor

    def sample(self, clusters):
        """
        :param int-array clusters: (n,)

        :return: coo-matrix of shape (n,n_tags)
        """
        n_entities = clusters.size
        n_clusters = int(numpy.rint(numpy.max(clusters))) + 1

        # clusters_tag_dim (n_clusters,) is the number of tags associated 
        # with each clusters
        if self.clusters_tag_dim is None:
            base_dim = int(numpy.sqrt(n_entities / n_clusters))
            clusters_tag_dim = base_dim*numpy.ones(n_clusters, dtype=int)
        else:
            clusters_tag_dim = self.clusters_tag_dim
        if numpy.isscalar(clusters_tag_dim):
            clusters_tag_dim = [clusters_tag_dim]*n_clusters
        clusters_tag_dim = numpy.array(clusters_tag_dim)

        # entities have a (more or less) random number of tags
        # depending on the cluster they belong to.
        # if unbalanced_factor is high, entities some cluster will have much more tags than others.
        unbalance_n_tags = ((clusters+1)/n_clusters *
                            self.n_tags_range[1] * self.unbalanced_factor).astype(int)
        entities_n_tags = (numpy.random.randint(*self.n_tags_range, size=clusters.size)
                           + numpy.random.randint(0, numpy.maximum(1, unbalance_n_tags)))

        clusters_tag_offset = numpy.cumsum(clusters_tag_dim) - clusters_tag_dim

        entities = numpy.repeat(numpy.arange(clusters.size), entities_n_tags)
        entities_cluster = clusters[entities]

        # sample the tags
        tags = (clusters_tag_offset[entities_cluster] + 
            numpy.random.rand(entities_n_tags.sum())*clusters_tag_dim[entities_cluster])
        tags = numpy.floor(tags).astype(int)

        avg_n_tags = entities_n_tags.mean()
        # abritrary suffle proportion 
        shuffle_proportion = numpy.power(self.difficulty, 0.5 + 1/(1 + avg_n_tags))
        # if `avg_n_tags` == 1 then `shuffle_proportion` = `self.difficulty`

        # shuffle
        shuffle = numpy.random.rand(tags.size) < shuffle_proportion
        shuffled_tags = tags[shuffle]
        numpy.random.shuffle(shuffled_tags)
        tags[shuffle] = shuffled_tags

        return to_structured([
            ('from', entities),
            ('to', tags.astype(self.DTYPE))
        ])


class CategoriesSampler(TagsSampler):
    NAME = 'category'
    IS_M2M = False

    def __init__(self, dims=1, difficulty=0):
        super().__init__(
            clusters_tag_dim=dims,
            n_tags_range=(1, 2),
            difficulty=difficulty,
            unbalanced_factor=0,
        )

    def sample(self, clusters):
        m2m = super().sample(clusters)
        return m2m['to']


class ScalarSampler(BaseFeatureSampler):
    NAME = 'scalar'
    IS_M2M = False
    DTYPE = 'f4'

    def sample(self, embeddings_coord):
        f = embeddings_coord.copy()
        shuffle = numpy.random.rand(embeddings_coord.size) < self.difficulty
        shuffled = f[shuffle]
        numpy.random.shuffle(shuffled)
        f[shuffle] = shuffled
        return f


def assert_feature_truth_consistency(feature, truth):
    msg = f'{feature} is not compatible with the synthetic model'
    if truth.dtype.kind in 'iu':
        assert isinstance(feature, TagsSampler), msg
    elif truth.dtype.kind == 'f':
        assert isinstance(feature, ScalarSampler), msg
    else:
        raise TypeError('unexpected users/items truth dtype: {truth.dtype}')


def sample_features(features, features_of, synthetic_truth):
    """
    :param list-of-subclass-of-BaseFeatureSampler features:
    :param str feature_of: either 'user' or 'item'
    :param array synthetic_truth:
    :return: tuple(
        array features_array: concatenation of non-m2m features in one single structured array
        list-of-dict features_m2ms: list of dict
            {'name': feature_name, 'array': struct-array['user/item_index', 'value_id']}
            (same format as what is exepected by the crossing minds API)
        )
    """
    assert features_of in ['user', 'item']
    assert len(features) <= synthetic_truth.shape[1], f'Too many {features_of} features.'

    features_data = []
    features_m2ms = []
    for i, (feature, synthetic_truth_slice) in enumerate(zip(features, synthetic_truth.T)):
        if feature is not None:
            assert_feature_truth_consistency(feature, synthetic_truth)
            ftr_vals = feature.sample(synthetic_truth_slice)
            ftr_name = f'f{i}_{feature.NAME}'
            if feature.IS_M2M:
                ftr_vals.dtype.names = f'{features_of}_index', 'value_id'
                features_m2ms.append({'name': ftr_name, 'array': ftr_vals})
            else:
                features_data.append((ftr_name, ftr_vals))
    features_array = to_structured(features_data) if features_data else None
    return features_array, features_m2ms