import os
import re
import json
import time
import traceback

from git import Repo
from xminds.lib.utils import deep_hash
from xminds.compat import logger

from apiclients import SYNTHETIC_DATASETS, ResultsDbConnector
from synthetic import SyntheticDataset
from recoapis import APIS, DummyRecoApi, RecommendationException, TrainingException
from utils import kwargs_product

EXP_NAME_PATTERN = re.compile(r'^[a-zA-Z\d][\w-]*')  # Amazon requirement applied to all
logger.setLevel('INFO')


def run_experiment(reco_api_class, exp_name, api_name,
                   dataset_hash, dataset, dataset_config,
                   config_test, api_config):
    """
    :returns: recos (arrays-of-int), meta, metrics, traceback_str
    """
    assert dataset_config['users_id_sample'], 'No users id to get reco from'
    # Meta
    repo = Repo('.')
    meta = {
        'repo_commit': str(repo.head.commit),
        'repo_untracked_files': json.dumps(repo.untracked_files),
    }

    # Prepare API
    api = reco_api_class(exp_name, dataset,
                         dataset_hash=dataset_hash,
                         environment=api_config['environment'])

    # Prepare resetting parameters
    traceback_str = ''

    # Check database is clean (fast if already empty)
    reset_check_time, _ = api.timed_reset()

    # Upload data
    upload_time, _ = api.timed_upload()

    # Train
    try:
        training_time, _ = api.timed_fit()
    except TrainingException:
        # Some training exceptions are expected. In this case we still mark the exp as DONE
        traceback_str = traceback.format_exc()
        recos = ([], [], [])
        t0 = time.time()
        api.reset()
        cleaning_time = time.time() - t0
        metrics = {
            'upload_time': upload_time,
            'reset_check_time': reset_check_time,
            'cleaning_time': cleaning_time,
        }
        return recos, meta, metrics, traceback_str

    # Get recos (with an optional delay to throttle requests)
    test_users_id = dataset_config['users_id_sample']
    try:
        testing_time, (users, items, rankings) = api.timed_recommend(
            test_users_id,  # dataset dependent
            exclude_rated_items=config_test.get('exclude_rated_items', True),
            n_items_per_user=config_test['n_items_per_user'],
            reco_delay=config_test.get('reco_delay', 0),
        )
        t_delay = len(test_users_id) * config_test.get('reco_delay', 0)
        testing_time = testing_time - t_delay  # remove delay from duration
    except RecommendationException:
        # Some reco exceptions are expected. In this case we still mark the exp as DONE
        users, items, rankings = [], [], []
        traceback_str = traceback.format_exc()
        testing_time = 0

    # Evaluate
    metrics = api.evaluate(users, items, rankings)

    # Clean DB/model/... that have just been used
    cleaning_time, _ = api.timed_reset()

    times = {
        'upload_time': upload_time,
        'training_time': training_time,
        'testing_time': testing_time,
        'cleaning_time': cleaning_time,
        'reset_check_time': reset_check_time,
    }
    metrics.update(times)
    recos = (users, items, rankings)
    logger.info(f'Finished. Times: {times}')
    return recos, meta, metrics, traceback_str


def parse_dataset_params(dataset_params):
    """
    Evalues 'py:*' patterns in the params
    """
    dim = dataset_params.get('dimension')  # some dims may be of the shape 'py:*', to `eval`
    if dim and isinstance(dim, str):
        if not dim.startswith('py:'):
            raise RuntimeError(f'Issue with dim={dim}')
        try:
            # literal_eval not working on tuple multiplication '(2,)*2'
            dataset_params['dimension'] = eval(dim[len('py:'):], {'__builtins__': None})
        except:
            logger.error(f'Issue interpreting {dim}')
            raise RuntimeError(f'Unexpected shape of {dim}')
    return dataset_params


def load_or_save(conn, dataset_params, synth_dt_cnf_hash, save_dataset_file, n_recos):
    '''
    Dataset and test users generated before the APIs loop to ensure all APIs use the same
    '''
    dataset_params = parse_dataset_params(dataset_params)

    dataset_file = os.path.join(SYNTHETIC_DATASETS, f'{synth_dt_cnf_hash}.json')
    if os.path.exists(dataset_file):
        # already saved
        logger.info(f'Loading from {synth_dt_cnf_hash}')
        dataset = SyntheticDataset.load(dataset_file)
        test_users_id = conn.get_users_id(synth_dt_cnf_hash)
        if not test_users_id:  # not saved
            logger.info('No test users ID loaded; creating new ones')
            test_users_id = DummyRecoApi.get_test_users(dataset, n_recos)  # Get users to test
        return dataset, test_users_id
    # not saved yet
    dataset = SyntheticDataset.sample(**dataset_params)
    test_users_id = DummyRecoApi.get_test_users(dataset, n_recos)  # Get users to test
    if save_dataset_file:
        dataset.save(dataset_file)
    return dataset, test_users_id


def run_qa(experiment, apis_str=None, force_run=None):
    """
    The QA as pseudo-code, with one for loop over dataset and one for loop over APIs

    for params in params_list:
        if database not saved for params exist
            create data for these params (items, users, ratings, parameters, test users)
            save it
        load data
        for api in apis:
            for api_config in api_configs:
                train model
                predict (get recos)
                evaluate recos
                save experiment/dataset/api configs and results

    :param dict experiment:
        {'experiment_name': str, 'save_file': str.pkl, 'parameters': dict, 'config_test': dict}
    :param str-or-list-of-str? apis_str: (default: None)
        - None or '_all' will run all APIs
        - One of or list from ['Amazon', 'Recombee', 'XMinds', 'Google', 'Dummy'] (case insensitive)
        - A coma-separated sublist of the above
    :param int? force_run: (default None) Provide a value to re-run an expe using `force_run` as
        exp_hash suffix.
    """
    if not apis_str or apis_str == '_all':
        apis_str = 'xminds,recombee,amazon,abacus'
    if ',' in apis_str:  # separate
        apis = [APIS[m.lower()] for m in apis_str.split(',')]
    else:
        apis = [apis_str] if isinstance(apis_str, str) else apis_str
        assert isinstance(apis, list)
        apis = [APIS[m.lower()] for m in apis]

    logger.info(f'Models to run: {", ".join([api.__name__ for api in apis])}')

    exp_name = experiment['experiment_name']  # may have to follow some api-specific rules
    assert EXP_NAME_PATTERN.match(exp_name), f'{exp_name} not matching pattern'
    api_configs = experiment['apis']
    environment = experiment['apis']['environment']
    test_config = experiment['test']
    gen_datasets_params = kwargs_product(**experiment['datasets'])
    save_dataset_file = experiment.get('save_datasets', True)
    n_recos = test_config.pop('n_recos')  # information kept in test_users_id
    result_db = ResultsDbConnector(environment)

    # DATASET
    for dataset_params in gen_datasets_params:
        synth_dt_cnf_hash = deep_hash(dataset_params, fmt='hex40')
        # dataset actually loaded later as it is heavy; better reloading a few times in a api loop
        # than loading every time we run the full `run.sh` script

        # API
        for RecoApi in apis:
            api_name = RecoApi.__name__
            short_api_name = re.sub('RecoApi', '', api_name)
            try:
                api_configs_product = kwargs_product(**api_configs[short_api_name])
            except KeyError:
                logger.debug(f'Default options running: {short_api_name} not in configs.')
                api_configs_product = ({},)
            for api_config in api_configs_product:
                api_config['environment'] = environment
                api_config['api_name'] = api_name
                api_hash = f'{api_name}{deep_hash(api_config, fmt="hex40")}'
                result_db.save_api(api_hash, api_config)

                # TEST (can be turned into `for` loop)
                test_hash = deep_hash(test_config, fmt='hex40')
                result_db.save_test(test_hash, test_config)

                # EXPERIMENT
                exp_hash = f'{exp_name}{api_hash}{test_hash}{synth_dt_cnf_hash}'
                if force_run:
                    # Allow rerunning a same expe (to assess code changes)
                    assert result_db.has_run_experiment(exp_hash), f'{exp_hash} must have run'
                    exp_hash = f'{exp_hash}#{force_run}'
                if result_db.has_run_experiment(exp_hash):
                    logger.info(f'Exp {exp_hash} already run or in progress. Skipping it. '
                                'You may `force` the run.')
                    continue
                if not result_db.can_run(api_name):
                    logger.info(f'We cannot run {api_name} now to avoid mismanagement. Skip.')
                    continue
                logger.info(f'api_config={api_config}')
                logger.info(f'test_config={test_config}')
                logger.info(f'api_hash={api_hash}')
                logger.info(f'experiment_hash = {exp_hash}')

                # Loading state (may be a bit slow)
                result_db.wait_if_being_created(synth_dt_cnf_hash)
                dataset, test_users_id = load_or_save(
                    result_db, dataset_params, synth_dt_cnf_hash, save_dataset_file, n_recos)
                dataset_config = dataset.get_config()
                logger.info(f'Synth.data.conf hash: {synth_dt_cnf_hash}; config: {dataset_config}')
                dataset_config['users_id_sample'] = test_users_id  # not a config but fits the use
                result_db.save_dataset(synth_dt_cnf_hash, dataset_config)

                # Run exp
                try:
                    result_db.mark_in_progress(exp_hash)
                    recos, meta, metrics, traceback_str = run_experiment(
                        RecoApi,
                        exp_name,
                        api_name,
                        synth_dt_cnf_hash,
                        dataset,
                        dataset_config,
                        test_config,
                        api_config,
                    )
                    # Save (in prod environment only)
                    result_db.save_recos(exp_hash, recos)
                    result_db.save_experiment(
                        exp_hash, exp_name, synth_dt_cnf_hash, test_hash, api_hash, meta,
                        metrics)
                except Exception:
                    logger.error(f'Exp {exp_hash} broke.')
                    traceback_str = traceback.format_exc()
                    result_db.mark_broken(exp_hash, traceback_str)
                    raise

                result_db.mark_done(exp_hash, traceback_str)
