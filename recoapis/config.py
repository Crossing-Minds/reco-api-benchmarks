from xminds.lib.utils import getenv


# ABACUS
# https://colab.research.google.com/github/abacusai/notebooks/blob/main/Use%20Cases/Personalized%20Recommendations%20Notebook.ipynb#scrollTo=sZZ0LDQ9vBsb
ABACUS_API_KEY = getenv('ABACUS_API_KEY', None)


# AMAZON
# https://stackoverflow.com/questions/21440709/how-do-i-get-aws-access-key-id-for-amazon/37947853#37947853
AWS_ROLE_ARN = getenv('AWS_API_BENCHMARK_ROLE_ARN', '')
AWS_BUCKET = getenv('AWS_API_BENCHMARK_BUCKET')
AWS_LOCAL_DATASET_DIR = getenv('AWS_API_BENCHMARK_LOCAL_DIR', 'data/amazon/')
AWS_SCHEMAS = {
    'users': {
        "type": "record",
        "name": "Users",
        "namespace": "com.amazonaws.personalize.schema",
        "fields": [
            {"name": "USER_ID", "type": "string"},
            {"name": "GENDER", "type": "string", "categorical": True}
        ],
        "version": "1.0"
    },
    'items': {
        "type": "record",
        "name": "Items",
        "namespace": "com.amazonaws.personalize.schema",
        "fields": [
            {"name": "ITEM_ID", "type": "string"},
            {"name": "GENRE", "type": "string", "categorical": True},
        ],
        "version": "1.0"
    },
    'explicit_interactions': {
        "type": "record",
        "name": "Items",
        "namespace": "com.amazonaws.personalize.schema",
        "fields": [
            {"name": "ITEM_ID", "type": "string"},
            {"name": "GENRE", "type": "string", "categorical": True},
        ],
        "version": "1.0"
    },
    'implicit_interactions': {
        "type": "record",
        "name": "Interactions",
        "namespace": "com.amazonaws.personalize.schema",
        "fields": [
            {"name": "USER_ID", "type": "string"},
            {"name": "ITEM_ID", "type": "string"},
            {"name": "EVENT_TYPE", "type": "string"},
            {"name": "TIMESTAMP", "type": "long"},
        ],
        "version": "1.0"
    }
}

# DUMMY
DUMMY_API_KEY = getenv('DUMMY_API_KEY', 'Hello world')


# GOOGLE
# https://cloud.google.com/recommendations-ai/docs/setting-up
GOOGLE_PROJECT_NUMBER = getenv('GOOGLE_API_PROJECT', None)
GOOGLE_API_KEY = getenv('GOOGLE_API_KEY', None)


# RECOMBEE
# https://docs.recombee.com/gettingstarted.html#manage-item-catalog
RECOMBEE_DBS_TOKENS = [[
        getenv(f'RECOMBEE_API_DB_ID{k}', None),  # identifier
        getenv(f'RECOMBEE_API_DB_TOKEN{k}', None),  # private token
    ] for k in range(10)]
RECOMBEE_DBS_TOKENS = [k for k in RECOMBEE_DBS_TOKENS if k[0]]  # remove missing slots


# CROSSING-MINDS
# https://docs.api.crossingminds.com/authentication.html#login-to-get-a-jwt
XMINDS_API_USER = getenv('XMINDS_API_B2B_BENCHMARK_USER', '')
XMINDS_API_PASSWORD = getenv('XMINDS_API_B2B_BENCHMARK_PASSWORD', '')
