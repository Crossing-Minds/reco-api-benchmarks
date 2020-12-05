import numpy

from xminds.ds.scaling import linearscaling

from .config import MIN_RATING, MAX_RATING
from .utils import DistributionTransport


class RatingsFactory:
    """
    This class articulates the synthetic model with the ratings scaler.
    Its main method is `get_ratings`
    
    In the future, it may also handle the ratings noise
    """
    N_SAMPLES_SCALER_FIT = 10_000

    def __init__(self, synthetic_model, scaler):
        """
        :param subclass-of-BaseSyntheticModel synthetic_model:
        :param subclass-of-BaseRatingsScaler scaler:
        """
        self.synthetic_model = synthetic_model
        self.scaler = scaler
        self._fit_scaler()

    @property
    def n_users(self):
        return self.synthetic_model.n_users

    @property
    def n_items(self):
        return self.synthetic_model.n_items

    def _fit_scaler(self):
        users_idx = numpy.random.randint(0, self.n_users, size=self.N_SAMPLES_SCALER_FIT)
        items_idx = numpy.random.randint(0, self.n_items, size=self.N_SAMPLES_SCALER_FIT)
        raw_ratings_sample = self.synthetic_model.get_ratings(users_idx, items_idx)
        self.scaler.fit(raw_ratings_sample)

    def get_ratings(self, users_idx, items_idx, scale=True):
        """
        :param (n,)-int-array users_idx:
        :param (n,)-int-array items_idx:
        :param bool? scale: whether to apply the scaler transform or not

        :returns: (n,)-float-array ratings
        """
        ratings = self.synthetic_model.get_ratings(users_idx, items_idx)
        if scale:
            ratings = self.scaler.transform(ratings)
        return ratings


class BaseRatingsScaler:
    """
    This class scales raw ratings produced by the ratings factory to 
    [MIN_TAING, MAX_RATING]. Subclasses implement different ways of doing that.
    """

    def fit(self, ratings):
        """
        :param 1d-float-array ratings:
        """
        raise NotImplementedError()

    def transform(self, ratings):
        """
        :param nd-float-array ratings:

        Map ratings into [MIN_RATING, MAX_RATING] (continuous interval)
        :return: nd-float-array scaled_ratings
        """
        shape = ratings.shape
        ratings = ratings.reshape(-1)
        ratings = self._transform(ratings)
        ratings = numpy.clip(ratings, MIN_RATING, MAX_RATING)
        return ratings.reshape(shape)

    def _transform(self, ratings):
        """
        :param 1d-float-array ratings: (n,)

        :return: 1d-float-array scaled_ratings (n,)
        """
        raise NotImplementedError()


class StandardRatingsScaler(BaseRatingsScaler):
    """
    Clip extreme ratings
    Then, scale the ratings linearly into [MIN_RATING, MAX_RATING]

    Clipping extreme values avoids that ratings are too concentrated 
    after the linear scaling step.
    """

    def __init__(self, q_clip=0.03):
        """
        :param float q: quantile for the clipping step

        Ratings will be clipped between the q-th quantile and the (1-q)-th quantile.
        """
        assert q_clip >= 0
        self.q_clip = q_clip

    def fit(self, ratings):
        self.q_low, self.q_high = numpy.quantile(ratings, [self.q_clip, 1-self.q_clip])
        return self

    def _transform(self, ratings):
        ratings = numpy.clip(ratings, self.q_low, self.q_high)
        ratings = linearscaling(ratings, MIN_RATING, MAX_RATING, self.q_low, self.q_high)
        return ratings


class GaussianRatingsScaler(BaseRatingsScaler):
    """
    Map the ratings to a gaussian distribution centered in 5.5 and restricted to [MIN_RATING, MAX_RATING].
    """

    def __init__(self):
        g = numpy.random.normal(loc=5.5, scale=2.2, size=10**4)
        self.gaussian_samples = g[(g > MIN_RATING) & (g < MAX_RATING)]
        self.mapper = DistributionTransport()

    def fit(self, ratings):
        self.mapper.fit(ratings, self.gaussian_samples)
        return self

    def _transform(self, ratings):
        return self.mapper.transform(ratings)
