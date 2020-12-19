import boto3

from ..config import AWS_ROLE_ARN, AWS_BUCKET, AWS_LOCAL_DATASET_DIR


class AmazonApiClients(object):
    ROLE_ARN = AWS_ROLE_ARN
    BUCKET = AWS_BUCKET
    LOCAL_DATASET_DIR = AWS_LOCAL_DATASET_DIR

    def __init__(self):
        self.personalize_client = boto3.client('personalize')
        self.events_client = boto3.client('personalize-events')
        self.runtime_client = boto3.client('personalize-runtime')
        self.s3_client = boto3.client('s3')
        self.logs_client = boto3.client('logs')
        # Checks
        assert self.BUCKET, 'Missing BUCKET; set varenv AWS_API_BENCHMARK_BUCKET'
        assert self.ROLE_ARN, 'Missing ROLE_ARN; set varenv AWS_API_BENCHMARK_ROLE_ARN'
