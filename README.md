# B2B API benchmark

This project evaluates the quality of various recommendation APIs. 

# Run 

To run all selected experiments (see below) on the XMinds API, 
```shell script
cd /path/to/reco-api-benchmarks/
sh script/run.sh xminds
```
All accepted APIs are (case insensitive)
```shell script
amazon|recombee|xminds|abacus
```
and each requires its own credentials (see the `APIs` section).

To run instead a single experiment `experiment_demo.yaml` on an XMinds API, 
```shell script
python . "experiments/configs/experiment_demo.yml" xminds
```

# Experiment

An experiment is defined as a `.yaml` file.  
- The experiment name (should follow the template `[a-zA-Z0-9][a-zA-Z0-9\-_]*`  for Amazon API) is 
an identifier to facilitate analysing the  results,
- The description should enable whoever reads it to know the raison d'être of the experiment,
- The `save_dataset` option is a boolean specifying whether to save the generated dataset locally,
- `apis.environment` specifies the environment. Should be `prod` for the real test. 
For all other values, results won't be saved in the results DB (to modify this behaviour, 
remove the `if`s in `apiclients.api_qa.Connector` methods). 
- The various `apis.${api}` and `datasets` are dicts containing values or list. 
These list will generate multiple runs, one for each possible combination. For example
for this experiment file
```yaml
experiment_name: 'experiment_file_demo'  
desciption: 'Exp file using various APIs parameters and dataset parameters'
save_datasets: True
apis:
  environment: 'prod'           # Will be saved in DB
  XMinds:
    algorithm:
      - 'default'               # default; optimised choice
      - 'knn'                   # Hypothetical algorithm name
  Recombee:
    algorithm:
      - 'recombee:personal'     # default
      - 'recombee:default'
  Amazon:
    algorithm:
      - 'hrnn'                  # legacy
      - 'user-personalization'  # default
    transform_to_implicit:
      - true                    # default
      - false
datasets:
  synthetic_model:              # Models of synthetic dataset
    - 'pure-clusters'
    - 'hierarchical-clustering'
  n_users:
    - 20_000
  n_items:
    - 2_000
  n_ratings:
    - 100_000
    - 1_000_000
  dimension:                    # Model-specific parameter
    - 5
    - 'py:(3,)*128'             # may be 'py:expr' (expr to be evaluated)
  ratings_scaling:
    - 'gaussian'
  interactions_distribution:
    - 'exponential'
  interactions_ratings_based:
    - true
  'users_features,items_features':   # If not-empty, will upload the features
    - [[], []]
    - [['cats1', 'tags2'], ['tags2', null, 'cats2', 'tags1', 'cats1']]  # (2 user features, 4 items)
test:
  n_items_per_user: 16          # Ask for 16 recommendation per user
  n_recos: 4_000                # Number of reco requests
```

there will be 4 different datasets (2 synthetic models * 2 n_ratings), and 9 API configurations 
(3 for XMinds, 2 for Recombee, 2 * 2 for Amazon), 
which would make a total of 9 * 4 = 36 pipeline runs, 
each collecting 4000 recos, hence the experiment would gather 144000 recommendations. 

# Scripts

- We've seen `scripts/run.sh` which runs all experiments on the provided (required) API. 
It can be run in parallel on different APIs with no issue, but there are conditions to run 
in parallel on the same API. This is allowed on XMinds and Amazon APIs but not on Recombee
as its databases are pre-defined. 

On top of `scripts/run.sh`, it's worthwhile to know how to use a few scripts. 

- Amazon API: `python scripts/reset_amazon.py` will clean all datasets, 
campaigns, solutions, schemas. May take a little while.
- `python scripts/print_sqlite_results.py` will print the tables of the results DB.
- `scripts/migration_into_api_table.py` was a one-time migration script to run point by point. 
Kept in case similar operation need be re-done. 

# APIs

Each API requires some credentials. 
## `XMinds`
Two environment variables should be provided: 
```
export XMINDS_API_B2B_BENCHMARK_USER='myname@my.domain'
export XMINDS_API_B2B_BENCHMARK_PASSWORD='abcdef153KriW8random'
```

## `Amazon`
Credentials and region are required for `boto3` to work:
```
# In ~/.aws/config
[default]
region=us-west-2   # if not too late

# In ~/.aws/credentials
[default]
aws_access_key_id = AKIA4QBGATH5132rrfrERrandom
aws_secret_access_key = 92bc83frG£432rF$%4/erfrandom
```
## `Recombee`

Manual work required: have an account, go to 
https://admin.recombee.com/databases/name_of_your_db/settings  (after providing the actual DB name), 
get the API identifier and the Private token. 
Then, you have a choice: only choose one DB, in which case setting the env var 
```
export RECOMBEE_API_DB_ID0='db1-prod'
export RECOMBEE_API_DB_TOKEN0='frT1fd4tbr2rSrandom'
...
export RECOMBEE_API_DB_ID4='db4-prod'
export RECOMBEE_API_DB_TOKEN4='ferwrw823fecrandom'
...
```
will be enough, or you wish to use a DB picked at random among a list of DBs, in which case  
fill the others (`RECOMBEE_API_DB_ID0`, ...). 
There are 2 DBs created by default, you can create more. 

The databases may bug and stay on delete mode, in which case having a manual control and 
at least a couple of DBs to use is comfortable. 


## `Abacus`
To use Abacus, you need an account. 
Log in and get the API key in https://abacus.ai/app/profile/apikey. 
Save it in the var env
```
export ABACUS_API_KEY='3rdede2323d4wdq33random'
```

# Datasets

Datasets are generated following the cross-product of the `.yaml` parameters. 
They are then saved so that following runs (using this API or another one) use the same dataset on the same. 

The filename uses a hash generated by `xminds.lib.util.deep_hash` 
on the generation parameters only (unlike `dataset.meta[‘hash’]` which hashes the data itself).

# Results

Define a `ROOT` directory in varenv  `XMINDS_API_QA_ROOT` (default: `'./data'`)
Results are saved in `RESULTS_DB=${XMINDS_API_QA_ROOT}results/results.sqlite`. 
