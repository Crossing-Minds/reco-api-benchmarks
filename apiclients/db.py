import re
import os
import numpy
import time
import json
import sqlite3

from xminds.compat import logger

from .config import BENCHMARKS_RESULTS_DB, BENCHMARK_DB_ATTRIBUTES


class ResultsDbConnector(object):
    """
    SQLite DB saving experiment parameters and results.
    Datasets with their ratings are saved elsewhere.
    In essence, the DB reproduces this structure:
        'status':   {experiment_hash: {'status': str, 'time': int, 'traceback': str}}
        'experiments':  {experiment_hash: {metric: json.dump, 'experiment_name': str,
                        'dataset_hash', 'api_hash', 'test_hash', 'meta'}}
        'datasets': {dataset_param_hash: {'dataset_config': json.dump, 'dataset_file': str}}
        'recos':    {experiment_hash: [(user_id, item_id, ranking)]
        'apis':     {api_hash: {'api_config': json.dump}}
        'tests':    {test_hash: {'test_config': json.dump, 'test_file': str}}
    """
    DB = BENCHMARKS_RESULTS_DB
    ATTRIBUTES = BENCHMARK_DB_ATTRIBUTES
    ATTRIBUTES_LIST = {key: [k[0] for k in values] for key, values in ATTRIBUTES.items()}

    def __init__(self, environment):
        """
        Creates a connector with the results DB and its tables:
            datasets, recos, configs, metrics, status, apis.
        Only in production environment will new results be saved
        :param str environment: 'staging', 'prod', 'local', or other
        """
        os.makedirs(os.path.dirname(self.DB), exist_ok=True)
        conn = sqlite3.connect(self.DB)

        def to_str(attrs):
            return ', '.join([k if isinstance(k, str) else ' '.join(
                [str(kk) for kk in k]) for k in attrs])
        c = conn.cursor()
        for table in self.ATTRIBUTES.keys():
            c.execute(f'''CREATE TABLE IF NOT EXISTS {table} ({to_str(self.ATTRIBUTES[table])})''')
        self.cursor = c
        self.conn = conn
        self.environment = environment

    def get_table_column_names(self, table):
        """
        :returns: list-of-str
        """
        cursor = self.cursor.execute(f'SELECT* FROM {table};')
        return [c[0] for c in cursor.description]

    def delete_column(self, table, column):
        """
        No method for deleting column in sqlite, requires workaround:
            - create new table as the one you are trying to change,
            - copy all data,
            - drop old table,
            - rename the new one.
        """
        assert table in self.ATTRIBUTES
        all_columns = self.get_table_column_names(table)
        assert column in all_columns, f'{column} missing from {all_columns}'
        attributes_to_save = [k for k in all_columns if k != column]
        attrs_str = ','.join(attributes_to_save)  # as str

        def to_str(attrs):
            return ', '.join([' '.join([str(kk) for kk in k]) for k in attrs if k[0] != column])
        logger.info(f'Attributes to save: {attrs_str}')
        self.cursor.execute(f'''DROP TABLE IF EXISTS {table}_backup;''')
        self.cursor.execute(f'''CREATE TABLE {table}_backup ({to_str(self.ATTRIBUTES[table])});''')
        self.cursor.execute(f'''INSERT INTO {table}_backup SELECT {attrs_str} FROM {table};''')
        self.cursor.execute(f'''DROP TABLE {table};''')
        self.cursor.execute(f'''CREATE TABLE {table} ({to_str(self.ATTRIBUTES[table])});''')
        self.cursor.execute(f'''INSERT INTO {table} SELECT {attrs_str} FROM {table}_backup;''')
        self.cursor.execute(f'''DROP TABLE {table}_backup;''')
        self.conn.commit()

    def add_column(self, table, column, data_type):
        """
        Adds a new column in given table
        Example script:

            from apiclients.api_qa import Connector
            c = Connector('exploration')
            c.add_column('metrics', 'proportion_reco_from_training', 'real')
        acting like (other values)
            self.cursor.execute('ALTER TABLE status ADD traceback TEXT;')
            self.conn.commit()
        """
        self.cursor.execute(f'ALTER TABLE {table} ADD {column} {data_type};')
        self.conn.commit()

    def has_dataset(self, dataset_hash):
        if self.environment != 'prod':
            logger.info(f'No dataset in {self.environment} ')
            return False
        for _ in self.cursor.execute(
                f"SELECT * FROM datasets WHERE dataset_hash = '{dataset_hash}'"):
            return True
        return False

    def get_dataset_state(self, dataset_hash):
        """
        Get the dataset state from its hash, parsing `dimension` if it was saved as a string
        Returns None if not in prod environment
        """
        if self.environment != 'prod':
            logger.info(f'State None in {self.environment} ')
            return None
        datasets = list(self.cursor.execute(
            f"SELECT * FROM datasets WHERE dataset_hash = '{dataset_hash}'"))
        if datasets:
            assert len(datasets) == 1
            d = dict(zip(self.ATTRIBUTES_LIST['datasets'], datasets[0]))
            # dimension_str back to dimension if needed
            dim_str = d.pop('dimension_str')  # json.dumps of the non-int dim
            if dim_str:
                # replace dim by dim_str
                assert not d.get('dimension'), f'Dataset should not have dimension and dim_str: {d}'
                d['dimension'] = json.loads(dim_str)
            return d
        return None

    def get_users_id(self, dataset_hash):
        e = self.execute(
            f"SELECT dataset_config FROM datasets WHERE dataset_hash = '{dataset_hash}'")
        e = list(e)
        if not e or e == [('IN_PROGRESS',)]:
            return
        try:
            users_id_sample = json.loads(e[0][0])['users_id_sample']
        except:
            logger.error(f'Issue with e[0][0] where e={e}')
            raise
        assert isinstance(users_id_sample, list) and isinstance(users_id_sample[0], int), \
            f'{users_id_sample} should be a list of int'
        return users_id_sample

    def can_run(self, reco_api_name):
        """
        Check whether Recombee/Amazon/...  is already being used.
        Recombee abd Abacus cannot run in parallel because all expes use the same DB;
        XMinds and Amazon can; they should reset before running
        """
        if self.environment != 'prod':
            logger.info(f'Can always run in {self.environment} (responsibility of the user)')
            return True
        assert re.sub('RecoApi', '', reco_api_name) in (
            'Recombee', 'Amazon', 'XMinds', 'Abacus', 'Dummy')
        apis_can_run_in_parallel = ('AmazonRecoApi', 'XMindsRecoApi')
        # these APIs use different datasets for different expes
        if reco_api_name in apis_can_run_in_parallel:
            logger.info(f'API {reco_api_name} can run in parallel')
            return True
        # These APIs cannot run two expes at the same time (shared DB for example)
        exps_in_progress = list(
            self.execute(f"SELECT * FROM status WHERE status = 'IN_PROGRESS'"))
        for exp_hash, _, _, _ in exps_in_progress:
            if exp_hash.startswith(reco_api_name):
                logger.info(f'Exp {exp_hash} already running on API {reco_api_name}')
                return False
        return True

    def mark_broken(self, exp_hash, traceback_str):
        """
        When the run broke, save the traceback for future debugging
        :param str exp_hash:
        :param str? traceback_str:
        """
        self._mark_status_as(exp_hash, 'BROKEN', traceback_str=traceback_str)

    def mark_in_progress(self, exp_hash):
        """
        When the run broke, save the traceback for future debugging
        :param str exp_hash:
        :param str? traceback_str:
        """
        logger.info(f'Running experiment {exp_hash}')
        self._mark_status_as(exp_hash, 'IN_PROGRESS')

    def mark_done(self, exp_hash, traceback_str=''):
        """
        Use traceback_str in case something broke in an expected manner
        """
        self._mark_status_as(exp_hash, 'DONE', traceback_str=traceback_str)

    def _mark_status_as(self, exp_hash, status, traceback_str=''):
        if self.environment != 'prod':
            logger.info(f'Not marking statuses in {self.environment}')
            return
        assert status in ('DONE', 'IN_PROGRESS', 'BROKEN')
        logger.info(f'Marking exp {exp_hash} as {status}')
        self.cursor.execute(
            'INSERT OR REPLACE INTO status (experiment_hash,status,time,traceback) '
            'VALUES (?,?,?,?);',
            (exp_hash, status, int(time.time()), traceback_str))
        self.conn.commit()

    def delete_row(self, exp_hash, table='status'):
        """For manual use only"""
        if table in ('experiments', 'status', 'recos'):
            _hash = 'experiment_hash'
        elif table == 'datasets':
            _hash = 'dataset_hash'
        elif table == 'apis':
            _hash = 'api_hash'
        elif table == 'tests':
            _hash = 'test_hash'
        else:
            raise NotImplementedError(table)
        self.cursor.execute(f"DELETE FROM {table} WHERE {_hash} = '{exp_hash}'")
        self.conn.commit()

    def has_run_experiment(self, exp_hash):
        """
        Dictates whether an experiment should be run:
        - when there is no metric saved in `metric` table for this hash,
        - when it's been IN_PROGRESS for over a day.
        Otherwise returns False
        :returns bool:
        """
        if self.environment != 'prod':
            logger.info(f'Always running experiments in {self.environment}')
            return False
        exps = list(self.cursor.execute(
            f"SELECT * FROM experiments WHERE experiment_hash = '{exp_hash}'"))
        if exps:
            # exp run, metrics saved: pass
            return True
        # check if it's just currently running
        exps = list(self.cursor.execute(
            f"SELECT * FROM status WHERE experiment_hash = '{exp_hash}'"))
        if not exps:
            # exp not run, not in progress: to do
            return False
        # if it's running, check status is IN_PROGRESS (if not, there's something weird)
        _, status, status_time, traceback_str = exps[0]
        traceback_str = traceback_str or '""'  # None by default, breaks json.loads
        if status == 'DONE':
            # should be True, but then the metrics were saved;
            # this allows to only delete metrics to rerun
            return False
        elif status == 'BROKEN':
            # Should be done now
            logger.info(f'Exp broke with traceback {traceback_str}')
            logger.info('Rerunning the exp.')
            return False
        elif status == 'IN_PROGRESS':
            pass  # undecided
        else:
            raise RuntimeError(exps[0])
        # if it's been 'in progress' for too long, we consider the run broke and re-run it
        delta = time.time() - status_time
        if delta > 24*60*60:
            logger.warning(f'Exp {exp_hash} being marked IN_PROGRESS for over a day. We re-run it')
            return False
        logger.info(f'Exp {exp_hash} is has been running for {delta} seconds; skip it')
        return True

    def save_recos(self, experiment_hash, recos):
        """
        Saves recos (3 arrays-or-tuple-or-list-of-int)
        :param recos: (array-of-int, array-of-int, array-of-int)  (users_id, items_id, rankings)
        """
        if isinstance(recos[0], numpy.ndarray):
            tolist = lambda l: l.tolist()
        elif isinstance(recos[0], (tuple, list)):
            tolist = lambda l: l
        else:
            raise NotImplementedError(type(recos[0]))
        _recos = (experiment_hash, *tuple(json.dumps(tolist(l)) for l in recos))
        self.cursor.execute(f"REPLACE INTO recos VALUES (?,?,?,?);", _recos)
        if self.environment != 'prod':
            logger.info(f'Only commiting in prod')
            return
        self.commit()

    def save_api(self, api_hash, api_config):
        self._save('apis', api_hash,json.dumps(api_config))

    def save_test(self, test_hash, test_config):
        self._save('tests', test_hash,json.dumps(test_config))

    def save_dataset(self, dataset_hash, dataset_config):
        self._save('datasets', dataset_hash, json.dumps(dataset_config))

    def wait_if_being_created(self, dataset_hash):
        """If 2 processes want to create/save the same dataset in parallel, the second one waits"""
        t0 = time.time()
        while True:
            duration = time.time() - t0
            configs = list(self.execute(
                f"SELECT dataset_config FROM datasets WHERE dataset_hash = '{dataset_hash}'"))
            if duration > 20 or not configs:
                if duration > 20:
                    logger.warning(f'Creating dataset ({dataset_hash}) should not take so long')
                # no one working on it, marking it as IN_PROGRESS as current process will do it
                self._save('datasets', dataset_hash, 'IN_PROGRESS')
                return
            if configs[0][0] == 'IN_PROGRESS':
                logger.info(f'Waiting for dataset {dataset_hash} to be saved...')
                time.sleep(5)
            else:
                # dataset ready to be loaded
                return

    def save_experiment(self, exp_hash, exp_name, dataset_hash, test_hash, api_hash, meta, metrics):
        """
        Only save in prod environment
        """
        meta_d = json.dumps(meta)
        metrics_d = json.dumps(metrics)
        self._save('experiments', exp_hash, exp_name, dataset_hash,
                   test_hash, api_hash, meta_d, metrics_d)

    def _save(self, table, *args):
        """
        Only save in prod environment
        args should be of the desired type and order
        """
        assert table in self.ATTRIBUTES
        self.execute(f"REPLACE INTO {table} VALUES ({','.join('?' for _ in args)});", args)
        if self.environment != 'prod':
            logger.info(f'Only commiting in prod')
            return
        self.commit()

    def print_table(self, table, head=None, tail=None):
        """
        Print the rows of given table.
        If `head` given, only print `head` first rows.
        If `tail` given, only print `tail` last rows.
        :param str table:
        :param int? head:
        :param int? tail:
        """
        print(self.get_table_column_names(table))
        ls = list(self.execute(f"SELECT * FROM {table}"))
        if not ls:
            print(f' -- table {table} empty --')
            return
        if head is not None:
            ls = ls[:head]
        if tail is not None:
            ls = ls[tail:]
        for line in ls:
            if table == 'datasets':
                config_str = line[1]
                try:
                    config = json.loads(config_str)
                except:
                    config = config_str
                    print(line[0], config)
                    continue
                users_id = config["users_id_sample"]
                if users_id is None:
                    print(f' --- ERROR: dataset_hash={line[0]} has a None users_id')
                else:
                    config['users_id_sample'] = f'{users_id[:10]}... (total: {len(users_id)}))'
                line = (line[0], config)
            elif table == 'recos':
                line = (line[0], *tuple(f'{k[:20]}...' for k in line[1:]))  # don't show all ids
            print(line)

    def print_tables(self):
        for k in ('datasets', 'apis', 'tests', 'experiments', 'status'):
            print(f'-- {k} -- ')
            self.print_table(k)
            print('')

    def execute(self, *kwargs):  # commit needed after
        return self.cursor.execute(*kwargs)

    def commit(self):
        self.conn.commit()


def print_all_tables():
    """
    Print all tables
    """
    c = ResultsDbConnector('print')
    c.print_tables()


if __name__ == '__main__':
    print_all_tables()
