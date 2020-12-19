import os
import shutil
import time
import numpy
from tqdm import tqdm

from abacusai.client import ApiClient, ApiException
from xminds.lib.arrays import set_or_add_to_structured
from xminds.lib.utils import retry
from xminds.compat import logger

from .baserecoapi import BaseRecoApi, RecommendationException, TrainingException
from .config import ABACUS_API_KEY


class AbacusRecoApi(BaseRecoApi):
    RESOURCES_DIR = 'abacus/data/tmp/'  # change carefully: removed in reset with shutil.rmtree
    API_KEY = ABACUS_API_KEY
    PROJECT_NAME = 'Movie Recommendations'
    PROJECT_USE_CASE = 'USER_RECOMMENDATIONS'
    DEPLOYMENT_NAME = 'movie_deploy'

    def __init__(self, name, dataset, dataset_hash=None,
                 algorithm='time-linear', environment=None, training_config=None,
                 transform_to_implicit=None, transform_algorithm=None):
        super().__init__(name, dataset,
                         dataset_hash=dataset_hash, environment=environment,
                         algorithm=algorithm, transform_algorithm=transform_algorithm,
                         transform_to_implicit=transform_to_implicit)
        assert self.API_KEY, f'An API key is required for Abacus'
        self.client = ApiClient(self.API_KEY)
        self.deployment_token = None  # defined during fit, used to get recos
        self.deployment_id = None
        self.items_dataset = None
        self.users_dataset = None
        self.ratings_dataset = None
        self.project = None
        self.training_config = training_config or {}

    def reset(self):
        shutil.rmtree(self.RESOURCES_DIR, ignore_errors=True)
        project = self._get_project(self.PROJECT_NAME)
        if project is None:
            return  # nothing to do

        deployments = project.list_deployments()
        for deploy in deployments:
            if deploy.name == self.DEPLOYMENT_NAME:
                self.client.delete_deployment(deploy.deployment_id)

        # deleting deployments may take a few seconds
        @retry(base=10, multiplier=1.2)
        def delete_project(project_id):
            self.client.delete_project(project_id)
            # delete_dataset needs dataset IDs; simpler to recreate project

        delete_project(project.project_id)

    def upload(self):
        files = self._prepare_resources(self.algorithm)
        project = self._create_or_get_project()
        self.project = project
        self._upload(project, files)

    def fit(self):
        self._validate_train_deploy(self.project)

    def recommend(self, test_user_ids, n_items_per_user=32, exclude_rated_items=True, reco_delay=0):
        """
        In `query_data`, if the key (`'user_id'`) is replaced by a name missing from columns,
        you'll receive constant recos. If the training was not good enough (possibly
        missing categorical features), the reco may break with `'user_id'` key.
        In this case, we consider this as an expected error and will skip the run.
        :param test_user_ids: test's users IDs
        :param int? n_items_per_user:
        :param bool? exclude_rated_items: (default True) excludes already rated items
        :param float? reco_delay: set to positive to reduce server stress
        :raises: RecommendationException
        """
        reco_users = []
        reco_items = []
        reco_data = []
        if self.deployment_token is None:
            raise NotImplementedError(f'deployment_token should be known')
        if self.deployment_id is None:
            raise NotImplementedError(f'deployment_id should be known')
        logger.info(f'Starting getting recos...')
        for user_id in tqdm(test_user_ids):
            query_data = {'user_id': str(int(user_id))}
            # `user_id` didn't work even though it appears on the website
            if exclude_rated_items:
                mask = self.dataset.ratings['user_id'] == user_id
                items = self.dataset.ratings['item_id'][mask]
                query_data.update({'excludeItemIds': items.astype('U64').tolist()})
            try:
                reco = self.client.get_recommendations(
                    self.deployment_token, self.deployment_id,
                    query_data, num_items=n_items_per_user,
                    page=1, include_filters=[], exclude_filters=[], score_field='')
            except ApiException as e:
                logger.error(f'Possibly the API requires categorical features. e={e}')
                raise RecommendationException(
                    'Abacus',
                    f'''Exception for query {query_data} on deployment 
                        token {self.deployment_token}, id {self.deployment_id}: {e}''')
            reco_users.extend([user_id] * len(reco))
            reco_items.extend([int(d['item_id']) for d in reco])
            reco_data.extend((len(reco) - numpy.arange(len(reco))).tolist())
        result = numpy.array(reco_users), numpy.array(reco_items), numpy.array(reco_data)
        return result

    def _create_or_get_project(self):
        """
        Create project on abacus server if doesn't exist; finds it otherwise
        :returns: Project
        """
        project = self._get_project(self.PROJECT_NAME)
        if project is not None:
            logger.info(f'Found {project}')
            return project
        project = self.client.create_project(name=self.PROJECT_NAME, use_case=self.PROJECT_USE_CASE)
        logger.info(f'Created {project}')
        return project

    def _get_project(self, project_name):
        """
        :returns: Project or None
        """
        project = None
        project_list = self.client.list_projects()
        for project_ in project_list:
            if project_.name == project_name:
                project = project_
                break
        if project is None:
            logger.info(f'Did not find project `{project_name}` '
                        f'among {[p.name for p in project_list]}')
        return project

    def _prepare_resources(self, algorithm):
        """
        Prepares the dataset before upload.
        The API requires at least a features of user and item.
        If not provided in dataset, we upload a column of 1s.
        :param str algorithm:  specifies timestamp algorithm, can be
            'random' (between two set dates)
            'time-linear' (best for another API)
            'time-quadratic' (spread more than 'time-linear')
        :returns: {'items': str, 'users': str, 'ratings': str} paths to the three files
        :raises: NotImplementedError (algorithm not recognised)
        """
        timestamps = self.ratings_to_timestamps(self.dataset.ratings, algorithm)

        users, items = self.get_user_item_flat_properties(self.dataset)
        users_m2ms, items_m2ms = self.get_user_item_m2m_properties(self.dataset, asdict=True)
        user_properties = list(self.dataset.iget_users_properties(yield_id=True))
        item_properties = list(self.dataset.iget_items_properties(yield_id=True))
        if len(item_properties) < 2 and not items_m2ms:
            # at least a feature; seems necessary
            items = set_or_add_to_structured(items, [('i0', 1)])
            item_properties.append(
                {'property_name': 'i0', 'value_type': 'int32', 'repeated': False})
        if len(user_properties) < 2 and not users_m2ms:
            users = set_or_add_to_structured(users, [('u0', 1)])
            user_properties.append(
                {'property_name': 'u0', 'value_type': 'int32', 'repeated': False})
        all_users = self.get_all_features(users, users_m2ms)
        all_items = self.get_all_features(items, items_m2ms)
        ratings = set_or_add_to_structured(
            self.dataset.ratings[['user_id', 'item_id', 'rating']], [('timestamp', timestamps)])

        # Save users, items, ratings in three csv files
        _hash = f'{int(time.time())}'
        _dir = os.path.join(self.RESOURCES_DIR, _hash)
        os.makedirs(_dir, exist_ok=True)
        files = {
            'items': os.path.join(_dir, 'items.csv'),
            'users': os.path.join(_dir, 'users.csv'),
            'ratings': os.path.join(_dir, 'ratings.csv'),
        }

        i_names = all_items.dtype.names  # [k['property_name'] for k in item_properties]
        u_names = all_users.dtype.names  # [k['property_name'] for k in user_properties]
        assert len(i_names) >= 1 and len(u_names) >= 1, (
            f'AbacusAPI requires feature in both items & users. i: {i_names}, u: {u_names}')
        # '"' necessary when 1 column given (API bug)
        item_header = ','.join([f'"{p}"' for p in i_names])
        user_header = ','.join([f'"{p}"' for p in u_names])

        item_fmt = ','.join(
            [self.prop_to_flag(p, repeated_as_strings=True) for p in item_properties])
        user_fmt = ','.join(
            [self.prop_to_flag(p, repeated_as_strings=True) for p in user_properties])
        numpy.savetxt(files['items'], all_items, delimiter=',',
                      header=item_header, comments='', fmt=item_fmt)
        numpy.savetxt(files['users'], all_users, delimiter=',',
                      header=user_header, comments='', fmt=user_fmt)
        ratings['rating'] = ratings['rating'].astype('int64')
        numpy.savetxt(files['ratings'],
                      ratings[['user_id', 'item_id', 'rating', 'timestamp']],
                      delimiter=',',
                      header='"user_id","item_id","rating","timestamp"',
                      comments='', fmt='%d,%d,%d,%d')
        logger.info(f'Saved resources in {files}')
        return files

    def _upload(self, project, files):
        """
        Upload items/users/ratings files to chosen project
        :param Project project: output of self._get_project
        :param dict files: {'items': str, 'users': str, 'ratings': str};
            strings are relative path inside 'abacus/data/'
        """
        logger.info(f'Uploading files...')
        # ITEM DATASET
        movies = self.client.create_dataset_from_local_file(
            name='Movies Attributes',
            file_format='csv',
            project_id=project.project_id,
            dataset_type='CATALOG_ATTRIBUTES',
        )
        self.items_dataset = movies
        with open(files["items"]) as file:
            self.client.upload_file_part(dataset_upload_id=movies.dataset_upload_id,
                                         part_number=1, part_data=file)
        self.client.complete_upload(dataset_upload_id=movies.dataset_upload_id)

        # USER DATASET
        users = self.client.create_dataset_from_local_file(
            name='Users Attributes',
            file_format='csv',
            project_id=project.project_id,
            dataset_type='USER_ATTRIBUTES',
        )
        self.users_dataset = users
        with open(files["users"]) as file:
            self.client.upload_file_part(dataset_upload_id=users.dataset_upload_id,
                                         part_number=1, part_data=file)
        self.client.complete_upload(dataset_upload_id=users.dataset_upload_id)

        # RATINGS DATASET
        ratings = self.client.create_dataset_from_local_file(
            name='User Movie Ratings',
            file_format='csv',
            project_id=project.project_id,
            dataset_type='USER_ITEM_INTERACTIONS')
        self.ratings_dataset = ratings
        with open(files["ratings"]) as file:
            self.client.upload_file_part(dataset_upload_id=ratings.dataset_upload_id,
                                         part_number=1, part_data=file)
        self.client.complete_upload(dataset_upload_id=ratings.dataset_upload_id)

    def _validate_train_deploy(self, project, wait=True, data_type='Categorical'):
        """
        Validates a project and starts training. If `wait=True`,
        waits for the training to be finished
        Categorical gave poor prediction, tried Numerical.
        Didn't change anything, kept Numerical
        However `categorical` may be necessary to get recos. Switching back to Categorical
        """
        assert data_type in ('Categorical', 'Numerical', 'CategoricalList')  # there are others

        valid = False
        time_waited = 0
        while not valid:
            result = project.validate()  # Doesn't validate schema; unsure whether necessary
            valid = result.valid
            if valid:
                logger.info(f'Dataset ready.')
                break
            logger.info(f'Model not ready to be trained; expect ~60s. Sleep10. Validate={result}')
            if time_waited > 180:
                raise AssertionError(f'project not valid. Result={result}')
            time.sleep(10)
            time_waited += 10
        # setting some mappings (may be optional)
        self.client.set_column_mapping(project_id=self.project.project_id,
                                  dataset_id=self.ratings_dataset.dataset_id,
                                  column='item_id', column_mapping='ITEM_ID')

        self.client.set_column_mapping(project_id=self.project.project_id,
                                  dataset_id=self.ratings_dataset.dataset_id,
                                  column='user_id', column_mapping='USER_ID')

        self.client.set_column_mapping(project_id=self.project.project_id,
                                  dataset_id=self.ratings_dataset.dataset_id,
                                  column='timestamp', column_mapping='TIMESTAMP')
        project.set_column_data_type(self.ratings_dataset.dataset_id, 'rating', data_type)
        # items
        item_properties = {
            k['property_name']: k for k in self.dataset.iget_items_properties(yield_id=False)}
        items_schema = project.get_schema(self.items_dataset.dataset_id)
        for schema in items_schema:
            if schema.name == 'item_id':
                continue
            try:
                is_repeated = item_properties[schema.name]['repeated']
            except KeyError:
                assert schema.name == 'i0', f'{schema.name} unknown'
                is_repeated = False
            if is_repeated:
                _type = 'Categorical_List'
            else:
                _type = data_type
            project.set_column_data_type(self.items_dataset.dataset_id, schema.name, _type)
        # users
        user_properties = {
            k['property_name']: k for k in self.dataset.iget_users_properties(yield_id=False)}
        users_schema = project.get_schema(self.users_dataset.dataset_id)
        for schema in users_schema:
            if schema.name == 'user_id':
                continue
            try:
                is_repeated = user_properties[schema.name]['repeated']
            except KeyError:
                assert schema.name == 'u0', f'{schema.name} unknown'
                is_repeated = False
            if is_repeated:
                _type = 'Categorical_List'
            else:
                _type = data_type
            project.set_column_data_type(self.users_dataset.dataset_id, schema.name, _type)
        logger.info(f'Starting training with configs: {self.project.get_training_config_options()}')
        model = project.train_model(training_config=self.training_config)
        # project.list_models()[0]  to recover it

        if wait:
            logger.info(f'Waiting for training to be finished, might take a while...')
            model.wait_for_evaluation()
            try:
                metrics = model.get_metrics()  # breaks with ApiException if training failed
            except ApiException as e:
                raise TrainingException('training', f'Training failed: {e}')
            logger.info(f'metrics={metrics}')
        logger.info('Deploying...')

        deployment = self.client.create_deployment(
            model_id=model.model_id, name=self.DEPLOYMENT_NAME,
            description='Movie reco', deployment_config={})
        deployment.wait_for_deployment(timeout=1000)
        self.deployment_token = project.create_deployment_token().deployment_token
        self.deployment_id = deployment.deployment_id
        return model

    @classmethod
    def prop_to_flag(cls, prop, repeated_as_strings=False):
        """
        Return the flag corresponding to the property's data kind
        :param dict prop: {'repeated': bool, 'value_type': numpy type}
        :param bool? repeated_as_strings: (default False) if set to True, return '%s' for repeated
        """
        if repeated_as_strings and prop['repeated']:
            return '%s'
        kind = cls.prop_to_kind(prop)
        if kind in 'iu':
            return '%d'
        if kind == 'f':
            return '%f'
        if kind == 'U':
            return '%s'
        raise NotImplementedError(kind)
