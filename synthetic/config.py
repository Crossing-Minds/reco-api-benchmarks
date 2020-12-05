import numpy

# For interaction samplers
INTERACTIONS_DTYPE = [('user', 'uint32'), ('item', 'uint32')]
RATINGS_DTYPE = [('user', 'uint32'), ('item', 'uint32'), ('rating', 'float32')]

MIN_RATING = 1. 
MAX_RATING = 10.

# For synthetic datasets
EXPLICIT_RATINGS_DISTRIBUTION = numpy.array([
    0.02444988, 0.0195599, 0.02689487, 0.06112469, 0.12224939,
    0.18337408, 0.19559902, 0.17114914, 0.12224939, 0.07334963
])  # arbitrary, hand-crafted.
IMPLICIT_RATINGS_DISTRIBUTION = numpy.array([
    0, 0, 0, 0, 0.05,
    0.2, 0.18, 0.15, 0.1, 0.06
])  # arbitrary, hand-crafted.
