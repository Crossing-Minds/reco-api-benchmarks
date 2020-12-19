import os
import yaml
from optparse import OptionParser

from apiclients import SYNTHETIC_DATASETS
from .api_qa import run_qa


def check_experiment(experiment_file, experiment):
    """
    Various checks
    """
    # assert experiment_file = 'exp_name.*'
    exp_name = experiment['experiment_name']
    assert os.path.basename(experiment_file).startswith(exp_name), (
        f'Experiment name `{exp_name}` is not the beginning of filename `{experiment_file}`')


def main():
    """
    # To run single parameter on XMinds API
    python apiclients/api_qa.py experiments/configs/debug.yml XMinds

    # Run dummy API (fast, debug mode)
    python apiclients/api_qa.py experiments/configs/debug.yml dummy

    # Run trivial exp on Amazon API and force the rerun (-f or --force)
    python apiclients/api_qa.py experiments/configs/trivial.yml Amazon -f

    # Run full experiment
    python apiclients/api_qa.py experiments/configs/experiment.yml
    """
    os.makedirs(SYNTHETIC_DATASETS, exist_ok=True)
    parser = OptionParser()

    parser.add_option("-f", "--force", dest="force",
                      help="Forces an expe to be re-run with `force` suffix")
    (options, args) = parser.parse_args()
    try:
        experiment_file = args[0]  # ex: r'experiment1.yml'
    except:
        raise RuntimeError('Provide a .yml experiment file')

    # logger.info(f'Experiment file: {experiment_file}')
    with open(experiment_file) as file:
        try:
            experiment = yaml.load(file, Loader=yaml.FullLoader)  # local;
        except (ValueError, AttributeError):
            experiment = yaml.load(file)  # on server

    try:
        models_str = args[1]
    except:
        models_str = ''

    force = options.force
    force = int(force) if force is not None else None
    run_qa(experiment, models_str, force_run=force)


if __name__ == '__main__':
    main()
