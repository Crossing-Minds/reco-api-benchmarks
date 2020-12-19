import numpy
from scipy.optimize import bisect

from xminds.compat import logger

from .config import INTERACTIONS_DTYPE, MIN_RATING, MAX_RATING
from .utils import partition_int, njit


class InteractionsSampler:
    """
    This class samples interactions, i.e. pairs user-item for which the ratings is known.
    Sampling interactions can equivalently be viewed as sampling a mask for the ratings matrix.
    Sampling is designed to be in O(n_interactions).
    """
    BUFFER_SIZE_MULTIPIER = 2.5
    ALLOWED_DISTRIBUTION_SCHEME = {'uniform', 'exponential', 'invlog'}
    N_SAMPLES_RATINGS_DIS_ESTIMATION = 300_000
    DEFAULT_ITEM_MAX_POPULARITY_FACTOR = 100

    def __init__(self, density, users_distribution='uniform', items_distribution='uniform',
                 min_per_user=None, max_per_user=None, max_item_popularity_factor=None,
                 ensure_one_per_item=True, target_ratings_distribution=None):
        """
        :param float density: must be in ]0, 1[
        :param string users_distribution: interactions distribution scheme for users
        :param string items_distribution: interactions distribution scheme for items
        :param int? min_per_user: minimum number of interactions per user
            (will be strictly respected)
        :param int? max_per_user: maximum number of interactions per user
        :param float? max_item_popularity_factor:
            = popularity(most popular item) / popularity(least popular item)
            (aimed but not strictly respected)
        :param bool? ensure_one_per_item: if True (default), each item will have at least
            one interaction
        :param float-array? target_ratings_distribution: array of size (MAX_RATING - MIN_RATING + 1)
            The ratings distribution to be aimed while sampling interactions for MNAR sampling.

        Users/Items distribution scheme can be either:
            - 'uniform': the number of interactions of each user/item will roughly be the same.
            - 'exponential': the number of interactions of users/items follow an
                exponential distribution
            - 'invlog': the number of interactions of users/items is distributed even more unevenly
                than for 'exponential'

        Note: MNAR means "missing not at random sampling"
            In recommendations, this term is usually used about the phenomenon:
                "a user has more chance to interact with an item he/she likes"
                (so the missing interactions are missing not at random)
        """
        assert 0 < density < 1
        assert users_distribution in self.ALLOWED_DISTRIBUTION_SCHEME
        assert items_distribution in self.ALLOWED_DISTRIBUTION_SCHEME
        self.density = density
        self.users_reparition = users_distribution
        self.items_reparition = items_distribution
        self.min_per_user = min_per_user
        self.max_per_user = max_per_user
        self.max_item_popularity_factor = max_item_popularity_factor
        self.ensure_one_per_item = ensure_one_per_item
        self.target_ratings_distribution = target_ratings_distribution

    def sample(self, n_users, n_items, ratings_factory=None):
        """
        :param int n_users:
        :param int n_items:
        :param RatingsFactoryBase? ratings_factory: Must be provided for MNAR sampling
        :returns: INTERACTIONS_DTYPE-array interactions
        """
        n_interacts = round(self.density * n_users * n_items)
        users_n_interacts = self._pick_users_n_interacts(
            n_users, n_items, n_interacts,
            self.users_reparition, self.min_per_user, self.max_per_user)
        items_popularity = self._pick_items_popularity(n_items, self.items_reparition,
                                                       self.max_item_popularity_factor)
        ratings_acceptance, bins, remaining_mass = None, None, None
        if self.target_ratings_distribution is not None:
            assert ratings_factory is not None, 'For MNAR sampling, must provide ratings factory'
            ratings_acceptance, bins, remaining_mass = self._compute_ratings_acceptance(
                n_users, n_items, ratings_factory, self.target_ratings_distribution)
        interacts, offset = self._sample_interactions(
            users_n_interacts, items_popularity, ratings_factory,
            ratings_acceptance=ratings_acceptance, ratings_bin_edges=bins,
            remaining_mass=remaining_mass)
        if self.ensure_one_per_item:
            interacts, offset = self._ensure_at_least_one_per_item(
                interacts, offset, n_users, n_items)
        users_n_interacts = numpy.bincount(interacts['user'][:offset])
        items_n_interacts = numpy.bincount(interacts['item'][:offset])
        nu_all = (users_n_interacts == n_items).sum()
        ni_all = (items_n_interacts == n_users).sum()
        if nu_all > 0:
            logger.warning(f'WARNING: some users ({nu_all:,}) have interactions with all the items')
        if ni_all > 0:
            logger.warning(f'WARNING: some items ({ni_all:,}) have interactions with all the users')
        return interacts[:offset]

    @classmethod
    def _sample_interactions(cls, users_n_interacts, items_popularity, ratings_factory,
                             ratings_acceptance=None, ratings_bin_edges=None, remaining_mass=None):
        """
        :param int-array users_n_interacts: (nu,)
        :param float-array items_popularity: (ni,)
        :param RatingsFactory ratings_factory: not used for missing at random sampling
            (i.e. not ratings-based interactions sampling)
        :param float-array? ratings_acceptance: (m,)
        :param float-array? ratings_bin_edges: (m+1,)
            `ratings_acceptance` and `ratings_bin_edges` are provided for MNAR sampling
                (i.e. ratings-based interactions sampling).
            They define the probabilty of keeping a sampled interaction given its rating value.
        :param float? remaining_mass:
        :returns: INTERACTIONS_DTYPE-array interactions, int
        """
        numpy.random.shuffle(users_n_interacts)
        numpy.random.shuffle(items_popularity)
        n_interacts = users_n_interacts.sum()
        n_items = items_popularity.size
        items_cp = items_popularity.cumsum()
        items_cp /= items_cp[-1]
        interacts = cls._init_interactions_buffer(n_interacts)
        items_mask = numpy.ones(n_items, dtype=bool)
        if ratings_acceptance is None:
            k = 0
            for u, dk in enumerate(users_n_interacts):
                interacts['user'][k:k+dk] = u
                interacts['item'][k:k+dk] = cls._sample_available_items(dk, items_cp, items_mask)
                k += dk
            return interacts, k
        else:
            k = 0
            mul = (1 / remaining_mass)*1.1 + 5
            max_n_tries = 10  # arbitrary value
            for u in range(users_n_interacts.size):
                dk = users_n_interacts[u]
                interacts['user'][k:k+dk] = u
                n_to_sample = min(n_items // 2, int(dk*mul))
                u_repeated = numpy.full(n_to_sample, u)
                for _ in range(max_n_tries):
                    items = cls._sample_available_items(n_to_sample, items_cp, items_mask)
                    # apply the missing not a random step
                    # i.e. keep interactions with a probability depending on their rating value
                    ratings = ratings_factory.get_ratings(u_repeated[:items.size], items)
                    idxs = numpy.searchsorted(ratings_bin_edges, ratings)
                    idxs = numpy.maximum(idxs - 1, 0)  # avoids issue when rating=MIN_RATING
                    keep_propability = ratings_acceptance[idxs]
                    keep = numpy.random.rand(items.size) < keep_propability
                    items = items[keep]
                    items = items[:dk]  # keep at most `dk` items
                    interacts['item'][k:k+items.size] = items
                    k += items.size
                    dk -= items.size
                    items_mask[items] = False
                    if dk == 0:
                        break
                if dk > 0:
                    raise ValueError(
                        f'Could not sampled {users_n_interacts[u]} interactions for one user')
                u_n_inters = users_n_interacts[u]
                items_mask[interacts['item'][k-u_n_inters:k]] = True
            return interacts, k

    @classmethod
    def _init_interactions_buffer(cls, n_interactions):
        """
        :param int n_interactions:
        :returns: interactions_buffer
        """
        buffer_size = int(n_interactions * cls.BUFFER_SIZE_MULTIPIER)
        interactions_buffer = numpy.empty(buffer_size, dtype=INTERACTIONS_DTYPE)
        return interactions_buffer

    @classmethod
    def _compute_ratings_acceptance(cls, n_users, n_items, ratings_factory, target_dis):
        """
        :param int n_users:
        :param int n_items:
        :param RatingsFactory ratings_factory:
        :param float-array target_dis: array of size MAX_RATING - MIN_RATING + 1. 
            The ratings distribution to be aimed while sampling interactions
        :returns: tuple(
            (n,)-float-array acceptance: a rating in the i-th bin will be kept
                with probability acceptance[i]
            (n+1,)-float-array bin_edges: edges of the bins (same as what is returned
                by `numpy.histogram`)
        )
        """
        rnd_users = numpy.random.choice(n_users, cls.N_SAMPLES_RATINGS_DIS_ESTIMATION)
        rnd_items = numpy.random.choice(n_items, cls.N_SAMPLES_RATINGS_DIS_ESTIMATION)
        ratings = ratings_factory.get_ratings(rnd_users, rnd_items)
        hist, bin_edges = numpy.histogram(ratings, bins=30, range=(MIN_RATING, MAX_RATING))
        acceptance = numpy.zeros(hist.size)
        for i, (n_rtgs_bin, left, right) in enumerate(zip(hist, bin_edges, bin_edges[1:])):
            if n_rtgs_bin > 0:
                mid = (left + right)/2 - MIN_RATING
                mid_floor = numpy.floor(mid).astype(int)
                mid_frac = mid - mid_floor
                target_val = (target_dis[mid_floor]*(1 - mid_frac)
                              + target_dis[mid_floor + 1]*mid_frac)
                acceptance[i] = target_val / n_rtgs_bin
        acceptance /= acceptance.max()
        remaining_mass = ((hist * acceptance) / hist.sum()).sum()
        if remaining_mass < 1/10:
            logger.warning('WARNING: Interactions sampling might be slow or even'
                           + f'impossible (remaining mass is {remaining_mass})')
        return acceptance, bin_edges, remaining_mass

    @classmethod
    def _ensure_at_least_one_per_item(cls, interacts_buffer, offset, n_users, n_items):
        """
        :param INTERACTIONS_DTYPE-array interacts_buffer:
        :param int offset: offset of the buffer (i.e. number of interactions in the buffer)
        :param int n_users:
        :param int n_items:
        """
        items_count = numpy.bincount(interacts_buffer['item'][:offset], minlength=n_items)
        items_no_interacts, = numpy.where(items_count == 0)
        n_no_interacts = items_no_interacts.size
        assert interacts_buffer.size >= offset + n_no_interacts, \
            f'interactions buffer too small: {interacts_buffer.size} < {offset} + {n_no_interacts}'

        interacts_buffer['item'][offset:offset + n_no_interacts] = items_no_interacts
        interacts_buffer['user'][offset:offset +
                                 n_no_interacts] = numpy.random.choice(n_users, n_no_interacts)
        return interacts_buffer, offset + n_no_interacts

    def _pick_users_n_interacts(self, n_users, n_items, n_interacts, scheme, min_interact=None,
                                max_interact=None):
        """
        :param int n_users:
        :param int n_items:
        :param int n_interacts: the desired number of interactions to sample (and thus to be
            distributed among users)
        :param str scheme: distribution scheme
        :param int? min_interacts:
        :param int? max_interacts:
        :returns: int-array (nu,) n_interacts_per_user
        """

        if scheme == 'uniform':
            msg = 'can not specify min_interact or max_interact for uniform distribution'
            assert min_interact is None and max_interact is None, msg
            return partition_int(n_interacts, n_users)
        elif scheme == 'exponential':
            min_interact = min_interact or 1
            assert max_interact is None, 'can not specify max_interact for exponential distribution'

            def compute_n_interacts(mi):
                """
                :param int-or-float mi: maximum number of interactions allowed for one user 

                Compute the number of interactions per user with an exponential shape (uneven distribution)
                in function of the maximum number of interactions allowed for one user 
                and other global parameters: n_users, min_interact
                """
                x = numpy.linspace(numpy.log(min_interact), numpy.log(mi), num=n_users)
                n_inters = (numpy.exp(x) + 1e-5).astype(int)
                return n_inters

            # performs a bisection method to find a value for `mi` (max_interact)
            def f(mi): return compute_n_interacts(mi).sum() - n_interacts
            max_interact = bisect(f,  min_interact + 1, n_items)
            return compute_n_interacts(max_interact)

        elif scheme == 'invlog':
            min_interact = min_interact or 1
            max_interact = max_interact or min(n_items//4, n_interacts//n_users * 30)
            # note that default of max_interact is very arbitrary

            def compute_n_interacts(mul):
                """
                :param float mul:

                Compute the number of interactions per user with a "invlog" shape (very uneven distribution)
                in function of the parameter `mul`, and other global parameters: n_users, max_interact, min_interact
                """
                x = numpy.linspace(0, 1, num=n_users + 1)[1:]
                eps = 1 / max_interact
                y = - 1 / (-eps + mul*numpy.log(x))
                n_inters = (y + min_interact).astype(int)
                return n_inters

            # performs a bisection method to find a value for `mul`
            # we are looking for a value such that `compute_n_interacts(mul)` ~= `n_interacts`
            # because we can't directly find it by solving an equation
            def f(mul): return compute_n_interacts(mul).sum() - n_interacts
            mul = bisect(f,  1e-3, 100)
            return compute_n_interacts(mul)

    def _pick_items_popularity(self, n_items, scheme, max_popularity_factor=None):
        """
        :param int n_items:
        :param str scheme: distribution scheme
        :param int? max_popularity_factor:

        :returns: float-array (ni,) items_popularity
        """
        max_popularity_factor = max_popularity_factor or self.DEFAULT_ITEM_MAX_POPULARITY_FACTOR
        if scheme == 'uniform':
            return numpy.ones(n_items)
        elif scheme == 'exponential':
            x = numpy.linspace(0, numpy.log(max_popularity_factor), num=n_items)
            return numpy.exp(x)
        elif scheme == 'invlog':
            x = numpy.linspace(0, 1, num=n_items + 1)[1:]
            eps = 1 / max_popularity_factor
            y = - 1 / (-eps + numpy.log(x))
            return y + 1

    @staticmethod
    @njit
    def _sample_available_items(n_to_sample, cp, items_mask):
        """
        :param int n_to_sample:
        :param (ni,)-float-array cp: cumulative distribution 
        :param (ni,)-bool-array items_mask:

        :returns: (n_to_sample,)-int-array sampled_items
        """
        sampled_items = numpy.empty(n_to_sample, dtype=numpy.int32)
        for k in range(n_to_sample):
            item = -1
            while item == -1 or not items_mask[item]:
                item = numpy.searchsorted(cp, numpy.random.rand())
            items_mask[item] = False
            sampled_items[k] = item
        items_mask[sampled_items] = True
        return sampled_items
