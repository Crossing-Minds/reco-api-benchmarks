import os
import numpy
import time
import json

from xminds.compat import logger

from ..baserecoapi import BaseRecoApi, JobFailedException, S3Exception
from ..config import AWS_SCHEMAS, AWS_LOCAL_DATASET_DIR


class AmazonPrepareApi(object):

    LOCAL_DIR = AWS_LOCAL_DATASET_DIR

    def __init__(self, api_clients):
        os.makedirs(api_clients.LOCAL_DATASET_DIR, exist_ok=True)
        self.clients = api_clients

    def aws_prepare_resources(self, arn_set, all_user_features, all_item_features, dataset,
                              name, transform_to_implicit, transform_algorithm):

        user_feature_type = 'float'
        item_feature_type = 'float'
        assert user_feature_type in ('float', 'int', 'string')
        assert item_feature_type in ('float', 'int', 'string')
        logger.info('aws_prepare_resources...')

        user_properties = list(dataset.iget_users_properties(yield_id=False))
        item_properties = list(dataset.iget_items_properties(yield_id=False))

        # create users' schema
        logger.info(f'user_properties={user_properties}')
        if not user_properties:
            user_schema = AWS_SCHEMAS['users']
        else:
            fields = [{
                "name": "USER_ID",
                "type": "string"
            }]
            fields.extend(
                [{
                    "name": prop['property_name'],
                    "type": self._type_to_type(prop) if not prop['repeated'] else 'string',
                    "categorical": prop['repeated'],
                } for prop in user_properties]
            )
            user_schema = {
                "type": "record",
                "name": "Users",
                "namespace": "com.amazonaws.personalize.schema",
                "fields": fields,
                "version": "1.0"
            }
        arn_set.users_schema_arn = self._aws_create_schema(
            user_schema,
            name=f'sch_u_{name}'
        )
        logger.info('users features schema created')
        # create items' schema
        logger.info(f'item_properties={item_properties}')
        if not item_properties:
            item_schema = AWS_SCHEMAS['items']
        else:
            fields = [{
                "name": "ITEM_ID",
                "type": "string"
            }]
            fields.extend(
                [{
                    "name": prop['property_name'],
                    "type": self._type_to_type(prop) if not prop['repeated'] else 'string',
                    "categorical": prop['repeated'],
                } for prop in item_properties]
            )
            item_schema = {
                "type": "record",
                "name": "Items",
                "namespace": "com.amazonaws.personalize.schema",
                "fields": fields,
                "version": "1.0"
            }
        arn_set.items_schema_arn = self._aws_create_schema(
            item_schema,
            name=f'sch_i_{name}'
        )
        logger.info('items features schema created')
        prefix_schema_interaction = 'implicit' if transform_to_implicit else 'explicit'
        prefix_schema_interaction_short = prefix_schema_interaction[0]  # i / e
        arn_set.interactions_schema_arn = self._aws_create_schema(
            AWS_SCHEMAS[prefix_schema_interaction + '_interactions'],
            name=f'sch_{prefix_schema_interaction_short}ints_{name}'
            # name includes i/e for robustness
        )
        arn_set.dataset_group_arn = self._aws_create_dataset_group(
            name=f'dset_gp_{name}'
        )
        if user_properties:
            logger.info('creating users dataset')
            arn_set.dataset_users_arn = self._aws_create_dataset_features(
                all_user_features, user_properties,
                arn_set.users_schema_arn,
                arn_set.dataset_group_arn,
                name=f'dset_u_{name}',  # dataset_users
                schema_type='USERS'
            )
            arn_set.dataset_import_job_arn = self._aws_import_dataset(
                arn_set.dataset_users_arn,
                name=f'dset_u_{name}'
            )
        if item_properties:
            logger.info('creating items dataset')
            arn_set.dataset_items_arn = self._aws_create_dataset_features(
                all_item_features, item_properties,
                arn_set.items_schema_arn,
                arn_set.dataset_group_arn,
                name=f'dset_i_{name}',
                schema_type='ITEMS'
            )
            arn_set.dataset_import_job_arn = self._aws_import_dataset(
                arn_set.dataset_items_arn,
                name=f'dset_i_{name}'
            )
        t0 = time.time()
        arn_set.dataset_arn = self._aws_create_dataset(
            dataset,
            transform_to_implicit,
            transform_algorithm,
            arn_set.interactions_schema_arn,
            arn_set.dataset_group_arn,
            name=f'dset_ints_{name}'  # dataset interactions
        )
        logger.info(f'Creating dataset took {time.time()-t0} seconds')
        t0 = time.time()
        arn_set.dataset_import_job_arn = self._aws_import_dataset(
            arn_set.dataset_arn,
            name=f'dset_ints_{name}'
        )
        logger.info(f'Importing dataset took {time.time()-t0} seconds')

    @staticmethod
    def _type_to_type(prop):
        kind = numpy.dtype(prop['value_type']).kind
        if kind in 'iu':
            return 'int'
        if kind == 'f':
            return 'float'
        if kind == 'U':
            return 'string'
        raise NotImplementedError(prop)

    def _aws_create_schema(self, data, name):
        if isinstance(data, dict):
            schema = json.dumps(data)
        else:
            with open(data) as file:
                schema = file.read()
        try:
            create_schema_response = self.clients.personalize_client.create_schema(
                name=name,
                schema=schema
            )
        except self.clients.personalize_client.exceptions.ResourceAlreadyExistsException:
            resp = self.clients.personalize_client.list_schemas()
            schemas = [s for s in resp['schemas'] if s['name'] == name]
            if len(schemas) != 1:
                raise RuntimeError(f'Cannot find the one schema ARN: resp={resp}')
            create_schema_response = schemas[0]
            logger.info(f'Schema already exists, but found its arn')
        schema_arn = create_schema_response['schemaArn']
        logger.info('Schema ARN: %s', schema_arn)
        return schema_arn

    def _aws_create_dataset_group(self, name):
        try:
            response = self.clients.personalize_client.create_dataset_group(name=name)
        except self.clients.personalize_client.exceptions.ResourceAlreadyExistsException:
            resp = self.clients.personalize_client.list_dataset_groups()
            dataset_groups = [s for s in resp['datasetGroups'] if s['name'] == name]
            if len(dataset_groups) != 1:
                raise RuntimeError(f'Cannot find the one dataset ARN: resp={resp}')
            response = dataset_groups[0]
            logger.info(f'Dataset group already existed, but found its arn')
        dataset_group_arn = response['datasetGroupArn']
        logger.info('Dataset group ARN: %s', dataset_group_arn)
        # wait for the dataset_group to be activated
        while True:
            description = self.clients.personalize_client.describe_dataset_group(
                datasetGroupArn=dataset_group_arn
            )
            status = description["datasetGroup"]["status"]
            logger.info('DatasetGroup: %s', status)
            if status == "ACTIVE":
                return dataset_group_arn
            if status == "CREATE FAILED":
                raise JobFailedException('dataset_group', description)
            logger.info(f'status={status}; trying again in 20 seconds...')
            time.sleep(20)

    def _aws_create_dataset(self, dataset, transform_to_implicit, transform_algorithm,
                            interactions_schema_arn, dataset_group_arn, name):
        # save dataset as csv
        logger.info('Transform explicit into implicit: %s', transform_to_implicit)
        filename = os.path.join(self.clients.LOCAL_DATASET_DIR, f'{name}_for_amazon.csv')
        if transform_to_implicit is False:  # information in `event_value` instead of `timestamp`
            ratings = dataset.ratings
            header = 'USER_ID,ITEM_ID,EVENT_TYPE,EVENT_VALUE,TIMESTAMP'
            # Dangerous to use a `view`: dtype.descr is not defined for types with overlapping
            # or out-of-order fields. Better directly define the right order, defined in the schema
            dtype_dict = {v[0]: v for v in ratings.dtype.descr + [
                ('event_type', '<U6'), ('timestamp', numpy.uint32)]}
            dtype_list = [dtype_dict[k] for k in
                          ('user_id', 'item_id', 'event_type', 'rating', 'timestamp')]
            new_dt = numpy.dtype(dtype_list)
            new_ratings = numpy.empty(ratings.shape, dtype=new_dt)
            new_ratings[['user_id', 'item_id', 'rating']] = ratings[['user_id', 'item_id','rating']]
            new_ratings['event_type'] = 'rating'
            new_ratings['timestamp'] = 0
            numpy.savetxt(filename,
                          new_ratings,  # [['user', 'item', 'event_type', 'rating', 'timestamp']],
                          delimiter=',',
                          header=header,
                          comments='',
                          fmt='%s')
        else:  # information in `timestamp` instead of `event_value`
            header = 'USER_ID,ITEM_ID,EVENT_TYPE,TIMESTAMP'
            ratings = BaseRecoApi.explicit_to_implicit(dataset, transform_algorithm)
            numpy.savetxt(filename,
                          ratings[['user_id', 'item_id', 'event_type', 'timestamp']],
                          delimiter=',',
                          header=header,
                          comments='',
                          fmt='%s')
        # save the file in s3
        try:
            self.clients.s3_client.upload_file(
                filename,
                self.clients.BUCKET,
                f'{name}.csv'
            )
        except Exception as ex:
            raise S3Exception('create_dataset', ex)
        # create the dataset inside the dataset_group
        try:
            response = self.clients.personalize_client.create_dataset(
                name=name,
                schemaArn=interactions_schema_arn,
                datasetGroupArn=dataset_group_arn,
                datasetType='INTERACTIONS'
            )
        except self.clients.personalize_client.exceptions.ResourceAlreadyExistsException:
            resp = self.clients.personalize_client.list_datasets()
            dataset_groups = [s for s in resp['datasets'] if s['name'] == name]
            if len(dataset_groups) != 1:
                raise RuntimeError(f'Cannot find the one dataset ARN: resp={resp}')
            response = dataset_groups[0]
            logger.info(f'Dataset already existed, but found its arn')
        logger.info('Dataset Arn: %s', response['datasetArn'])
        return response['datasetArn']

    def _aws_create_dataset_features(self, all_features, properties, schema_arn,
                                     dataset_group_arn, name, schema_type):
        """Will modify the dtype.names"""
        # save dataset as csv
        filename = os.path.join(self.clients.LOCAL_DATASET_DIR,
                                f'{name}_{schema_type}_for_amazon.csv')
        # capital ID name expected
        all_features.dtype.names = tuple(n.upper() if n in ('user_id', 'item_id') else n
                                         for n in all_features.dtype.names)
        header0 = 'USER_ID'if schema_type == 'USERS' else 'ITEM_ID'
        header1 = [p['property_name'] for p in properties]
        header = f'{header0},{",".join(header1)}'
        numpy.savetxt(filename,
                      all_features,
                      delimiter=',',
                      header=header,
                      comments='',
                      fmt='%s')
        # save the file in s3
        try:
            response = self.clients.s3_client.upload_file(
                filename,
                self.clients.BUCKET,
                f'{name}.csv'
            )
        except Exception as ex:
            raise S3Exception('create_dataset', ex)
        # create the dataset inside the dataset_group
        response = self.clients.personalize_client.create_dataset(
            name=name,
            schemaArn=schema_arn,
            datasetGroupArn=dataset_group_arn,
            datasetType=schema_type
        )
        logger.info('Dataset Arn: %s', response['datasetArn'])
        return response['datasetArn']

    def _aws_import_dataset(self, dataset_arn, name):
        """
        Of use for debugging, to check what's in the bucket
             import pandas as pd
             s3 = boto3.client('s3')
             obj = s3.get_object(Bucket=cls.BUCKET,
                Key='dset_u_debug_run_b78a9723d1a2e67_2ddd681a0c.csv')
             initial_df = pd.read_csv(obj['Body'])
        """
        # import the dataset inside personalize
        s3_filename = f's3://{self.clients.BUCKET}/{name}.csv'
        job_name = f'job_{name}'
        try:
            response = self.clients.personalize_client.create_dataset_import_job(
                jobName=job_name,
                datasetArn=dataset_arn,
                dataSource={'dataLocation': s3_filename},
                roleArn=self.clients.ROLE_ARN)
        except self.clients.personalize_client.exceptions.ResourceAlreadyExistsException:
            resp = self.clients.personalize_client.list_dataset_import_jobs()
            dataset_import_jobs = [s for s in resp['datasetImportJobs'] if s['jobName'] == job_name]
            if len(dataset_import_jobs) != 1:
                raise RuntimeError(f'Cannot find the one dataset import ARN: resp={resp}')
            response = dataset_import_jobs[0]
            logger.info(f'Dataset already existed, but found its arn')
        dsij_arn = response['datasetImportJobArn']
        logger.info('Dataset Import Job arn: %s', dsij_arn)
        # wait for the dataset_group to be activated
        while True:
            description = self.clients.personalize_client.describe_dataset_import_job(
                datasetImportJobArn=dsij_arn
            )
            status = description["datasetImportJob"]["status"]
            logger.info('DatasetImportJob: %s', status)
            if status == "ACTIVE":
                return dsij_arn
            if status == "CREATE FAILED":
                raise JobFailedException('import_dataset', description)
            time.sleep(20)  # may take 10 minutes+


def aws_prepare_resources(api_clients, *args):
    AmazonPrepareApi(api_clients).aws_prepare_resources(*args)
