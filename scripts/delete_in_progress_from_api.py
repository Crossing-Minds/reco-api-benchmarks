# Delete where status = IN_PROGRESS for given API. Make sure there is no run from this API currently
# running or there may be lost results or broken runs

import sys
from apiclients.db import ResultsDbConnector


def delete_in_progress_status(api):
    real_api_names = {
        'xminds': 'XMindsRecoApi',
        'dummy': 'DummyRecoApi',
        'recombee': 'RecombeeRecoApi',
        'amazon': 'AmazonRecoApi',
        'abacus': 'AbacusRecoApi',
        'google': 'GoogleRecoApi',
    }
    api_name = real_api_names[api]
    c = ResultsDbConnector('reset_status')
    for exp_hash, status in c.execute('SELECT experiment_hash,status FROM status'):
        if api_name in exp_hash and status == 'IN_PROGRESS':
            print(f'Found {exp_hash} in progress. Deleting it from db')
            c.delete_row(exp_hash)


if __name__ == '__main__':
    api = sys.argv[1]
    delete_in_progress_status(api)
