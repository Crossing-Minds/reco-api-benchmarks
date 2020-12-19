import time
from xminds.compat import logger


class AmazonResetApiBase():
    """Base class to reset either the full bucket or just the """
    def __init__(self, arn_set, aws_personalize_client):
        self.arn_set = arn_set
        self.personalize_client = aws_personalize_client

    #     DELETE    methods
    def delete_filters(self, dataset_group_arn):
        for filter_ in list(self.ilist_filters(dataset_group_arn)):
            filter_arn = filter_['filterArn']
            logger.info(f'Deleting filter {filter_arn}')
            try:
                self.personalize_client.delete_filter(filterArn=filter_arn)
            except self.personalize_client.exceptions.ResourceNotFoundException:
                logger.info(f'Filter {filter_arn} was not found')
            except self.personalize_client.exceptions.ResourceNotFoundException as e:
                logger.info('Filter already deleted.')
                break

    def delete_datasets(self, dataset_group_arn):
        for dataset in list(self.ilist_datasets(dataset_group_arn)):
            dataset_arn = dataset['datasetArn']
            while True:
                logger.info(f'Deleting dataset {dataset_arn}')
                try:
                    self.personalize_client.delete_dataset(datasetArn=dataset_arn)
                    break
                except self.personalize_client.exceptions.ResourceInUseException as e:
                    logger.info(f'Resource in use, sleep(10)..., e={e}')
                    time.sleep(10)
                except self.personalize_client.exceptions.ResourceNotFoundException as e:
                    logger.info('Dataset already deleted.')
                    break

    def delete_solutions(self, dataset_group_arn):
        # delete campaigns first, then the solutions
        for solution in list(self.ilist_solutions(dataset_group_arn)):
            solution_arn = solution['solutionArn']
            for campaign in list(self.ilist_campaigns(solution_arn)):
                campaign_arn = campaign['campaignArn']
                while True:
                    logger.info(f'Deleting campaign {campaign_arn}')
                    try:
                        self.personalize_client.delete_campaign(campaignArn=campaign_arn)
                        break
                    except self.personalize_client.exceptions.ResourceInUseException:
                        time.sleep(10)
                    except self.personalize_client.exceptions.ResourceNotFoundException as e:
                        logger.info('Campaign already deleted.')
                        break
            while True:
                e = ''
                logger.info(f'Deleting solution {solution_arn} {e}')
                try:
                    self.personalize_client.delete_solution(solutionArn=solution_arn)
                    break
                except self.personalize_client.exceptions.ResourceInUseException as e:
                    time.sleep(10)
                except self.personalize_client.exceptions.ResourceNotFoundException as e:
                    logger.info('Solution already deleted.')
                    break

    def delete_dataset_group(self, dataset_group_arn):
        t0 = time.time()
        while True:
            logger.info(f'Deleting dataset_group {dataset_group_arn}')
            try:
                self.personalize_client.delete_dataset_group(datasetGroupArn=dataset_group_arn)
                time.sleep(5)
            except self.personalize_client.exceptions.ResourceInUseException as e:
                logger.info(f'Dataset group still in use... sleep(10), e={e}')
                if time.time() - t0 > 100:
                    # should not be useful, but may get stuck sometime without it
                    logger.info('Deleting a dataset may have failed; trying again')
                    self.delete_datasets(dataset_group_arn)
                time.sleep(10)
            except self.personalize_client.exceptions.ResourceNotFoundException:
                logger.info('Dataset groupe deleted.')
                break

    def delete_schemas(self):
        for schema in self.ilist_schemas():
            schema_arn = schema['schemaArn']
            while True:
                e = ''
                logger.info(f'Deleting schema {schema_arn} {e}')
                try:
                    self.personalize_client.delete_schema(schemaArn=schema_arn)
                    time.sleep(5)
                except self.personalize_client.exceptions.ResourceInUseException as e:
                    logger.info('Schema still in use... sleep(10)')
                    time.sleep(5)
                except self.personalize_client.exceptions.ResourceNotFoundException as e:
                    logger.info('Schema deleted.')
                    break

    def reset(self):
        logger.info('AWS Resetting...')
        for dataset_group in list(self.ilist_dataset_groups()):
            dataset_group_arn = dataset_group['datasetGroupArn']
            logger.info(f'Found dataset group {dataset_group_arn}. Delete what uses it first')
            self.delete_filters(dataset_group_arn)
            self.delete_datasets(dataset_group_arn)
            self.delete_solutions(dataset_group_arn)
            self.delete_dataset_group(dataset_group_arn)
        self.delete_schemas()
        logger.info('Finished resetting')

    def ilist_dataset_groups(self):
        raise NotImplementedError('Use subclass')

    def ilist_schemas(self):
        raise NotImplementedError('Use subclass')

    def ilist_solutions(self, dataset_group_arn):
        raise NotImplementedError('Use subclass')

    def ilist_campaigns(self, solution_arn):
        raise NotImplementedError('Use subclass')

    def ilist_datasets(self, dataset_group_arn):
        raise NotImplementedError('Use subclass')

    def ilist_filters(self, dataset_group_arn):
        raise NotImplementedError('Use subclass')


class AmazonResetApiAll(AmazonResetApiBase):
    # GET methods to delete all objects in the bucket
    def __init__(self, arn_set, aws_personalize_client):
        super().__init__(arn_set, aws_personalize_client)

    def ilist_dataset_groups(self):
        response = self.personalize_client.list_dataset_groups()
        dataset_groups = response['datasetGroups']
        while dataset_groups:
            for dataset_group in dataset_groups:
                yield dataset_group
            nextToken = response.get('nextToken')
            if not nextToken:
                break
            response = self.personalize_client.list_dataset_groups(nextToken=nextToken)
            dataset_groups = response['datasetGroups']

    def ilist_datasets(self, dataset_group_arn):
        response = self.personalize_client.list_datasets(datasetGroupArn=dataset_group_arn)
        datasets = response['datasets']
        while datasets:
            for dataset in datasets:
                yield dataset
            nextToken = response.get('nextToken')
            if not nextToken:
                break
            response = self.personalize_client.list_datasets(
                datasetGroupArn=dataset_group_arn, nextToken=nextToken)
            datasets = response['datasets']

    def ilist_solutions(self, dataset_group_arn):
        response = self.personalize_client.list_solutions(datasetGroupArn=dataset_group_arn)
        solutions = response['solutions']
        while solutions:
            for solution in solutions:
                yield solution
            nextToken = response.get('nextToken')
            if not nextToken:
                break
            response = self.personalize_client.list_solutions(
                datasetGroupArn=dataset_group_arn, nextToken=nextToken)
            solutions = response['solutions']

    def ilist_campaigns(self, solution_arn):
        response = self.personalize_client.list_campaigns(solutionArn=solution_arn)
        campaigns = response['campaigns']
        while campaigns:
            for campaign in campaigns:
                yield campaign
            nextToken = response.get('nextToken')
            if not nextToken:
                break
            response = self.personalize_client.list_campaigns(
                solutionArn=solution_arn, nextToken=nextToken)
            campaigns = response['campaigns']

    def ilist_schemas(self):
        response = self.personalize_client.list_schemas()
        schemas = response['schemas']
        while schemas:
            for schema in schemas:
                yield schema
            nextToken = response.get('nextToken')
            if not nextToken:
                break
            response = self.personalize_client.list_schemas(nextToken=nextToken)
            schemas = response['schemas']

    def ilist_filters(self, dataset_group_arn):
        response = self.personalize_client.list_filters(datasetGroupArn=dataset_group_arn)
        filters = response['Filters']
        while filters:
            for filter_ in filters:
                yield filter_
            nextToken = response.get('nextToken')
            if not nextToken:
                break
            response = self.personalize_client.list_filters(nextToken=nextToken)
            filters = response['Filters']


class AmazonResetApiOnlySelf(AmazonResetApiBase):
    # GET methods to delete only objects in the bucket from the current run
    def __init__(self, arn_set, aws_personalize_client):
        super().__init__(arn_set, aws_personalize_client)

    def ilist_dataset_groups(self):
        if self.arn_set.dataset_group_arn:
            yield {'datasetGroupArn': self.arn_set.dataset_group_arn}

    def ilist_datasets(self, dataset_group_arn):
        if self.arn_set.dataset_arn:
            yield {'datasetArn': self.arn_set.dataset_arn}

    def ilist_solutions(self, dataset_group_arn):
        if self.arn_set.solution_arn:
            yield {'solutionArn': self.arn_set.solution_arn}

    def ilist_campaigns(self, solution_arn):
        if self.arn_set.campaign_arn:
            yield {'campaignArn': self.arn_set.campaign_arn}

    def ilist_schemas(self):
        if self.arn_set.items_schema_arn:
            yield {'schemaArn': self.arn_set.items_schema_arn}
        if self.arn_set.users_schema_arn:
            yield {'schemaArn': self.arn_set.users_schema_arn}
        if self.arn_set.interactions_schema_arn:
            yield {'schemaArn': self.arn_set.interactions_schema_arn}

    def ilist_filters(self, dataset_group_arn):
        if self.arn_set.filter_arn:
            yield {'filterArn': self.arn_set.filter_arn}


def aws_reset(arn_set, only_self, aws_personalize_client):
    if only_self:
        resetter = AmazonResetApiOnlySelf(arn_set, aws_personalize_client)
    else:
        resetter = AmazonResetApiAll(arn_set, aws_personalize_client)
    resetter.reset()
