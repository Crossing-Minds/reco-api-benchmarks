import time
import numpy

from .baserecoapi import BaseRecoApi
from .config import DUMMY_API_KEY


class DummyRecoApi(BaseRecoApi):
    """Dummy api to run tests quickly"""

    def __init__(self, name, dataset, **kwargs):
        super().__init__(name, dataset, **kwargs)
        self.client = None  # Most APIs should define their client here
        assert DUMMY_API_KEY, "API's settings are checked at init time"

    def upload(self):
        pass

    def fit(self, algorithm=str, transform_to_implicit=bool, transform_algorithm=str):
        pass

    def recommend(self, test_user_ids, n_items_per_user=3, exclude_rated_items=True, reco_delay=0):
        # Assuming test_users_ids=[1,2] is used and user_id=2/item_id=3 was in training dataset
        reco_users = [1, 1, 1, 2, 2, 2]
        time.sleep(reco_delay)
        reco_items = [1, 2, 3, 1, 2, 4]
        reco_data = [3, 2, 1, 3, 2, 1]
        return numpy.array(reco_users), numpy.array(reco_items), numpy.array(reco_data)

    def reset(self):
        pass
