from apiclients.db import ResultsDbConnector
import sys


def delete():
    exp_hash = sys.argv[1]
    table = sys.argv[2]
    print(f'About to delete experiment {exp_hash} from table {table}...')
    c = ResultsDbConnector('exploration')
    c.delete_row(exp_hash, table)
    print('Done')


if __name__ == '__main__':
    delete()
