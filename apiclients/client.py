from xminds.api.client import CrossingMindsApiClient

from .config import ENVS_HOST


class ApiClientInternal(CrossingMindsApiClient):
    ENVS_HOST = ENVS_HOST

    def __init__(self, **kwargs):
        host = self.ENVS_HOST
        super().__init__(host=host, **kwargs)
