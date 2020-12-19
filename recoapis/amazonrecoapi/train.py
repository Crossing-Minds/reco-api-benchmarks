import time

from xminds.compat import logger

from ..baserecoapi import JobFailedException


class AmazonTrainApi():

    def __init__(self, api_clients):
        self.clients = api_clients

    def aws_train(self, arn_set, algorithm, event_type, name):

        t0 = time.time()
        arn_set.solution_arn = self._aws_create_solution(
            algorithm,
            event_type,
            arn_set.dataset_group_arn,
            name=f'solut_{name}'  # solution
        )
        logger.info(f'- Creating solution took {time.time()-t0} seconds')
        t0 = time.time()
        arn_set.solution_version_arn = self._aws_create_solution_version(
            arn_set.solution_arn
        )
        logger.info(f'Creating solution version took {time.time()-t0} seconds')
        # better to do it now, otherwise we'll have to wait later
        t0 = time.time()
        excl_rated_items = 'EXCLUDE itemId WHERE INTERACTIONS.event_type in ("click","rating")'
        response = self.clients.personalize_client.create_filter(
            name=f'exclude_{name}',
            datasetGroupArn=arn_set.dataset_group_arn,
            filterExpression=excl_rated_items
        )
        arn_set.filter_arn = response['filterArn']
        logger.info(f'Creating filter `{excl_rated_items}` took {time.time()-t0} seconds')

        t0 = time.time()
        arn_set.campaign_arn = self._aws_create_campaign(
            arn_set.solution_version_arn,
            name=f'campaign_{name}'
        )
        logger.info(f'Creating campaign took {time.time()-t0} seconds')

    def _aws_create_solution(self, algorithm, event_type, dataset_group_arn, name):
        # select amazon algorithm
        recipe_arn = f'arn:aws:personalize:::recipe/aws-{algorithm}'
        logger.info(f'create solution: recipe_arn={recipe_arn}, event_type={event_type}')
        # create solution
        try:
            create_solution_response = self.clients.personalize_client.create_solution(
                name=name,
                datasetGroupArn=dataset_group_arn,
                recipeArn=recipe_arn,
                eventType=event_type
            )
        except self.clients.personalize_client.exceptions.ResourceAlreadyExistsException:
            resp = self.clients.personalize_client.list_solutions()
            solutions = [s for s in resp['solutions'] if s['name'] == name]
            if len(solutions) != 1:
                raise RuntimeError(f'Cannot find the one solution: resp={resp}')
            create_solution_response = solutions[0]
            logger.info(f'Solution already existed, but found its arn.')
        solution_arn = create_solution_response['solutionArn']
        logger.info('Solution ARN: %s', solution_arn)
        return solution_arn

    def _aws_create_solution_version(self, solution_arn):
        try:
            solution_version_response = self.clients.personalize_client.create_solution_version(
                solutionArn=solution_arn
            )
        except self.clients.personalize_client.exceptions.ResourceAlreadyExistsException:
            resp = self.clients.personalize_client.list_solution_versions()
            versions = [s for s in resp['solutionVersions']
                        if s['solutionVersionArn'] == solution_arn]
            if len(versions) != 1:
                raise RuntimeError(f'Cannot find the one solution+version: resp={resp}')
            solution_version_response = versions[0]
            logger.info(f'Solution version already existed, but found its arn')
        solution_version_arn = solution_version_response['solutionVersionArn']
        logger.info('Solution version ARN: %s', solution_version_arn)
        # wait for the solution_version to be activated
        while True:
            description = self.clients.personalize_client.describe_solution_version(
                solutionVersionArn=solution_version_arn
            )
            status = description["solutionVersion"]["status"]
            logger.info("SolutionVersion: %s", status)
            if status == "ACTIVE":
                return solution_version_arn
            if status == "CREATE FAILED":
                raise JobFailedException('solution_version', description)
            time.sleep(20)

    def _aws_create_campaign(self, solution_version_arn, name):
        try:
            create_campaign_response = self.clients.personalize_client.create_campaign(
                name=name,
                solutionVersionArn=solution_version_arn,
                minProvisionedTPS=1
            )
        except self.clients.personalize_client.exceptions.ResourceAlreadyExistsException:
            resp = self.clients.personalize_client.list_campaigns()
            campaigns = [s for s in resp['campaigns'] if s['name'] == name]
            if len(campaigns) != 1:
                raise RuntimeError(f'Cannot find the one campaign: resp={resp}')
            create_campaign_response = campaigns[0]
            logger.info(f'Campaign already existed, but found its arn')
        campaign_arn = create_campaign_response['campaignArn']
        logger.info('Campaign ARN: %s', campaign_arn)
        # wait for the solution_version to be activated
        while True:
            description = self.clients.personalize_client.describe_campaign(
                campaignArn=campaign_arn
            )
            status = description["campaign"]["status"]
            logger.info('Campaign: %s', status)
            if status == "ACTIVE":
                return campaign_arn
            if status == "CREATE FAILED":
                raise JobFailedException('campaign', description)
            time.sleep(20)


def aws_train_resources(api_clients, *args):
    AmazonTrainApi(api_clients).aws_train(*args)
