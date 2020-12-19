from .baserecoapi import TrainingException, RecommendationException
from .abacusrecoapi import AbacusRecoApi
from .amazonrecoapi import AmazonRecoApi
from .dummyrecoapi import DummyRecoApi
from .recombeerecoapi import RecombeeRecoApi
from .xmindsrecoapi import XMindsRecoApi


APIS = {
    'recombee': RecombeeRecoApi,
    'xminds': XMindsRecoApi,
    'amazon': AmazonRecoApi,
    'abacus': AbacusRecoApi,
    'dummy': DummyRecoApi,
}

