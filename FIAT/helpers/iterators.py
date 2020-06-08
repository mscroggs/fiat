from itertools import product


def index_iterator(shp):
    """Constructs a generator iterating over all indices in
    shp in generalized column-major order  So if shp = (2,2), then we
    construct the sequence (0,0),(0,1),(1,0),(1,1)"""
    return product(*[range(i) for i in shp])
