from xminds.lib.utils import getenv


ROOT = getenv('XMINDS_API_QA_ROOT', './data')
SYNTHETIC_DATASETS = f'{ROOT}/synthetic/'
BENCHMARKS_RESULTS_DB = f'{ROOT}/results/results_2.sqlite'

# Hosts to log to, depending on the environment
ENVS_HOST = getenv('XMINDS_API_HOST', 'https://api.crossingminds.com')


# The DB's tables, their columns and types, with a few examples
# ORDER MATTERS: same order as in the db, do not change randomly
BENCHMARK_DB_ATTRIBUTES = {
    'status': (
        ('experiment_hash', 'text', 'UNIQUE'),
        ('status', 'text'),
        ('time', 'int'),
        ('traceback', 'text'),  # save dumps of traceback when breaks
    ),
    'experiments': (
        ('experiment_hash', 'text', 'UNIQUE'),
        ('experiment_name', 'text'),
        ('dataset_hash', 'text'),
        ('test_hash', 'text'),
        ('api_hash', 'text'),
        ('meta', 'text'),   # '{"repo_commit": "abc123", ...}'
        ('metrics', 'text')  # '{"mean_rating": 5.54, "dcg": 2.652}'
    ),
    'datasets': (
        ('dataset_hash', 'text', 'UNIQUE'),
        ('dataset_config', 'text'),  # dimension, ... users_id_sample for test
    ),
    'apis': (
        ('api_hash', 'text', 'UNIQUE'),
        ('api_config', 'text'),  # api_name, environment, algorithm, transform_to_implicit/algo
    ),
    'tests': (
        ('test_hash', 'text', 'UNIQUE'),
        ('test_config', 'text'),  # n_recos, n_items_per_user
    ),
    'recos': (
        ('experiment_hash', 'text', 'UNIQUE'),
        ('users_id', 'text'),  # json.dumps(users_array.tolist())
        ('items_id', 'text'),
        ('rankings', 'text')
    ),
}
