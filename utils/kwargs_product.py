from itertools import product

from xminds.lib.iterable import unzip


def kwargs_product(**kwargs):
    """
    The kwargs must be a mapping string->list.

    Example:
    >>> kwargs_product(
    ...     a=[1, 2],
    ...     b=['x', 'y'],
    ...     **{'c,d': [(0, 0), (1, 1)]}
    ... )
    [{'a': 1, 'b': 'x', 'c': 0, 'd': 0},
    {'a': 1, 'b': 'x', 'c': 1, 'd': 1},
    {'a': 1, 'b': 'y', 'c': 0, 'd': 0},
    {'a': 1, 'b': 'y', 'c': 1, 'd': 1},
    {'a': 2, 'b': 'x', 'c': 0, 'd': 0},
    {'a': 2, 'b': 'x', 'c': 1, 'd': 1},
    {'a': 2, 'b': 'y', 'c': 0, 'd': 0},
    {'a': 2, 'b': 'y', 'c': 1, 'd': 1}]

    :return list-of-dict:
    """
    args_name, args = unzip(kwargs.items())
    args_prod = product(*args)
    dicts = (dict(zip(args_name, args)) for args in args_prod)
    return [split_kwargs(d) for d in dicts]


def split_kwargs(kwargs):
    """
    :param dict kwargs:

    Split keys with commas and the corresponding values accordingly
    (values of comma-keys must be tuples of the same length than the splitted key).
    Example:
    >>> split_kwargs({'a,b': (1, 2), 'c': 3})
    {'a': 1, 'b': 2, 'c': 3}

    :return dict:
    """
    splitted_kwargs = {}
    for key, val in kwargs.items():
        if ',' in key:
            keys = key.split(',')
            assert len(val) == len(keys)
            for k, v in zip(keys, val):
                splitted_kwargs[k] = v
        else:
            splitted_kwargs[key] = val
    return splitted_kwargs

