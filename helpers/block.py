import numpy as np

"""
helper functions for spatial estimation
- distance matrix
- (semi-) variogram matrices

Thomas Pfaff & Claus Haslauer
2015
claus.haslauer@uni-tuebingen.de
"""


def calcBlockVariogramMatrix(blocks, variogram):
    """
    >>> nug = NuggetVariogram({'sill':3.})
    >>> exp = ExponentialVariogram({'sill':4.,'range':5.,})
    >>> cv = CompositeVariogram()
    >>> cv.addVariogram(nug)
    >>> cv.addVariogram(exp)
    >>> blockA = [[0.0, 0.0], [1.0, 0.0]]
    >>> blockB = [[0.0, 1.0], [1.0, 1.0]]
    >>> blocks = np.array([blockA, blockB])
    >>> calcBlockVariogramMatrix(blocks, cv)
    array([[ 0.        ,  3.85526186],
           [ 3.85526186,  0.        ]])
    """
    nblocks = len(blocks)
    variogramMatrix = np.zeros((nblocks, nblocks))
    for i in range(nblocks):
        for j in range(i):
            tmp = calcMeanVariogram(variogram, blocks[i], blocks[j])
            variogramMatrix[i, j] = tmp
            variogramMatrix[j, i] = tmp
    return variogramMatrix


def calcBlockCoVariogramMatrix(blocks, blocks_co, variogram):
    """
    """
    nblocks = blocks.shape[0]
    nblocks_co = blocks_co.shape[0]
    covariogramMatrix = np.zeros((nblocks, nblocks_co))
    for i in range(nblocks):
        for j in range(nblocks_co):
            tmp = calcMeanVariogram(variogram, blocks[i], blocks_co[j])
            covariogramMatrix[i, j] = tmp
    return covariogramMatrix


def calcDistanceMatrix(pointsA, pointsB):
    """function to calculate the euclidean distances between all points of two sets.
    pointsA and pointsB are (npoints x dimension) numpy arrays containing
    the cartesian coordinates of the points (one point per row)
    The number of points in pointsA and pointsB need not be the same.
    The result will be a (npointsA x npointsB)-numpy array

    Examples:
    in 2-D
    >>> pointsA = np.array([[0.0, 0.0], [1.0, 0.0]])
    >>> pointsB = np.array([[0.0, 1.0], [1.0, 1.0]])
    >>> calcDistanceMatrix(pointsA, pointsB)
    array([[ 1.        ,  1.41421356],
           [ 1.41421356,  1.        ]])

    in 3-D
    >>> pointsA = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    >>> pointsB = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    >>> calcDistanceMatrix(pointsA, pointsB)
    array([[ 1.        ,  1.41421356],
           [ 1.41421356,  1.        ]])

    With different numbers of points
    >>> pointsA = np.array([[0.0, 0.0]])
    >>> pointsB = np.array([[0.0,1.0], [1.0, 1.0]])
    >>> calcDistanceMatrix(pointsA, pointsB)
    array([[ 1.        ,  1.41421356]])

    Can also be used to calculate all distances within the same set of points
    >>> points = np.array([ [1.,1.],[-1.,-1.],[0.5,-1.1],[-0.8,0.9]])
    >>> calcDistanceMatrix(points, points)
    array([[ 0.        ,  2.82842712,  2.15870331,  1.80277564],
           [ 2.82842712,  0.        ,  1.50332964,  1.91049732],
           [ 2.15870331,  1.50332964,  0.        ,  2.38537209],
           [ 1.80277564,  1.91049732,  2.38537209,  0.        ]])

    Points should be of the same dimension
    >>> pointsA = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    >>> pointsB = np.array([[0.0, 1.0], [1.0, 1.0]])
    >>> calcDistanceMatrix(pointsA, pointsB)
    Traceback (most recent call last):
    ...
    AssertionError: Dimension of points has to be the same (dimA=3) != (2=dimB
    """
    # make sure the dimensions of both point sets are the same
    assert pointsA.shape[1] == pointsB.shape[1], \
        'Dimension of points has to be the same (dimA=%i) != (%i=dimB' % \
        (pointsA.shape[1], pointsB.shape[1])

    # set up some variables
    npointsA = pointsA.shape[0]
    npointsB = pointsB.shape[0]
    dimension = pointsA.shape[1]

    # initialize our result matrix and the helper vectors
    distmatrix = np.zeros((npointsA, npointsB))

    # for each dimension calculate the squared differences and add them to
    # the distance matrix
    for dim in range(dimension):
        # create numpy matrix for points A
        xxa = np.mat(np.ones((2, npointsA)))
        xxa[0, :] = pointsA[:, dim]

        # create numpy matrix for points B
        xxb = np.mat(np.ones((2, npointsB)) * -1)
        xxb[1, :] = pointsB[:, dim]

        # calculate distance matrix summand using matrix product
        # the squaring afterwards has to be done elementwise and therfore
        # the resulting matrix has to be converted to array before doing the
        # power operation
        distmatrix += np.asarray((xxa.T * xxb))**2

    # return the result
    return np.sqrt(distmatrix)


def calcMeanVariogram(variogram, blockA, blockB):
    """Calculates mean variogram values used for Block-Kriging
    >>> nug = NuggetVariogram({'sill':3.})
    >>> exp = ExponentialVariogram({'sill':4.,'range':5.,})
    >>> cv = CompositeVariogram()
    >>> cv.addVariogram(nug)
    >>> cv.addVariogram(exp)
    >>> pointsA = np.array([[0.0, 0.0], [1.0, 0.0]])
    >>> pointsB = np.array([[0.0, 1.0], [1.0, 1.0]])
    >>> calcMeanVariogram(cv, pointsA, pointsB)
    3.8552618609565066
    """
    if blockA.shape != blockB.shape:
        print('blockA.shape: ', blockA.shape)
        print('blockB.shape: ', blockB.shape)
        raise Exception
    else:
        return np.mean(variogram.calculate(calcDistanceMatrix(blockA, blockB)))


def calcCentroids(blocks):
    """
    >>> a = np.arange(5*4*3).reshape((5,4,3))
    >>> b = a.mean(axis=1)
    >>> b
    array([[  4.5,   5.5,   6.5],
           [ 16.5,  17.5,  18.5],
           [ 28.5,  29.5,  30.5],
           [ 40.5,  41.5,  42.5],
           [ 52.5,  53.5,  54.5]])
    """
    # this assumes blocks is 3-D (nblocks, npoints, ndims)
    return blocks.mean(axis=1)


def _test():
    import doctest
    print('running doctests on module')
    doctest.testmod()
    print('doctest finished')


if __name__ == '__main__':
    _test()