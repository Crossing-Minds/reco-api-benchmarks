from recoapis import AmazonRecoApi
from xminds.compat import logger

logger.setLevel('INFO')


def reset():
    print('Resetting Amazon AWS (all datasets/campaigns/...)')
    name, dataset = 'name', 'dataset'
    AmazonRecoApi(name, dataset).reset(only_self=False)


if __name__ == '__main__':
    reset()
