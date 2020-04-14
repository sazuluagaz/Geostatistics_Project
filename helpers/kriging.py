import numpy as np
import scipy.spatial

import helpers.block as block

import matplotlib.pyplot as plt
import scipy.spatial.distance as ssd
from configparser import SafeConfigParser
import helpers.variogram

"""
a python class for Kriging
  - OK: Ordinary Kriging
  - EDK: External Drift Kriging

  (Sebastian Gnann, 2016)
  - COK: Co-Kriging 
  - RCOK: Rescaled Co-Kriging
  - added function for choosing neighbouring points
  
Claus Haslauer and Thomas Pfaff
2015
claus.haslauer@uni-tuebingen.de
"""


def krigmatrix_ok(controls, variogram):
    """ Calculates the Kriging-Matrix for Ordinary Kriging.
    >>> vario = variogram.ExponentialVariogram()
    >>> vario.setParameters({'range':2,'sill':3,})
    >>> points = np.array([ [[1.,1.]],[[-1.,-1.]],[[0.5,-1.1]],[[-0.8,0.9]]])
    >>> krigmatrix_ok(points, vario)
    array([[ 0.        ,  2.2706498 ,  1.98055269,  1.78198258,  1.        ],
           [ 2.2706498 ,  0.        ,  1.58525759,  1.84585059,  1.        ],
           [ 1.98055269,  1.58525759,  0.        ,  2.08978437,  1.        ],
           [ 1.78198258,  1.84585059,  2.08978437,  0.        ,  1.        ],
           [ 1.        ,  1.        ,  1.        ,  1.        ,  0.        ]])
    """

    var_matrix = block.calcBlockVariogramMatrix(controls, variogram)

    # ok_matrix is the kriging matrix (which is being constructed here
    # add ones in the bottom row of the distance matrix
    horz_vec = [np.ones(len(controls))]
    glue_pt1 = np.vstack((var_matrix, horz_vec))
    # add on the right side a vector with an additional one
    vert_vec_a = np.transpose((horz_vec))
    vert_vec_b = np.vstack((vert_vec_a, [[0]]))
    ok_matrix = np.hstack((glue_pt1, vert_vec_b))

    return ok_matrix


def krigmatrix_edk(controls, control_extdrift, variogram):
    """ Calculates the Kriging-Matrix for External Drift Kriging.
    >>> vario = variogram.ExponentialVariogram()
    >>> vario.setParameters({'range':2,'sill':3,})
    >>> points = np.array([ [[1.,1.]],[[-1.,-1.]],[[0.5,-1.1]],[[-0.8,0.9]]])
    >>> krigmatrix_edk(points, np.array([4., 3., 5., 3.]), vario)
    array([[ 0.        ,  2.2706498 ,  1.98055269,  1.78198258,  1.        ,
             4.        ],
           [ 2.2706498 ,  0.        ,  1.58525759,  1.84585059,  1.        ,
             3.        ],
           [ 1.98055269,  1.58525759,  0.        ,  2.08978437,  1.        ,
             5.        ],
           [ 1.78198258,  1.84585059,  2.08978437,  0.        ,  1.        ,
             3.        ],
           [ 1.        ,  1.        ,  1.        ,  1.        ,  0.        ,
             0.        ],
           [ 4.        ,  3.        ,  5.        ,  3.        ,  0.        ,
             0.        ]])
    """
    # Zusammensetzen der EDK Matrix
    #  erster Baustein: OK Matrix
    #  daran unten den EDK Vektor mit einer Null ganz rechts bauen

    ok_matrix = krigmatrix_ok(controls, variogram)
    edk_a = np.hstack((control_extdrift.flatten(), [0]))
    edk_b = np.vstack((ok_matrix, edk_a))
    edk_c = np.hstack((edk_a, np.array([0.0])))
    edk_matrix = np.hstack((edk_b, edk_c[:, np.newaxis]))

    return edk_matrix


def krigmatrix_cok(controls, controls_co, 
                   variogram, variogram_co, crossvariogram):
    """ Calculates the Kriging-Matrix for Co-Kriging.
    Note: Only for two variables.
     
     1...n_u        n_u+1...n_u+n_v     n_u+n_v+1 

    [Gamma_uu       Gamma_uv            1 0]
    [                                   . .] 
    [Gamma_uv       Gamma_vv            0 1]
    [                                   . .]
    [1 .... 1       0 .... 0            0 0]
    [0 .... 0       1 .... 1            0 0]

    The Gammas are the (cross-)variogram matrices: 
    - Gamma_uu and Gamma_vv are the variograms of the two variables
    - Gamma_uv and Gamma_vu are the cross-variograms of the two variables (symmetric).
    Two additional rows/columns for the two constraints (two Lagrange multipliers).

    >>> vario = variogram.ExponentialVariogram()
    >>> vario.setParameters({'range':2,'sill':3,})
    >>> vario_co = variogram.ExponentialVariogram()
    >>> vario_co.setParameters({'range':2,'sill':3})
    >>> crossvario = variogram.ExponentialVariogram()
    >>> crossvario.setParameters({'range':2,'sill':3})
    >>> points = np.array([ [[1.,1.]],[[-1.,-1.]],[[0.,0.]] ])
    >>> points_co = np.array([ [[1.,1.]],[[-1.,-1.]],[[0.,0.]] ])
    >>> krigmatrix_cok(points, points_co, vario, vario_co, crossvario)
    array([[ 0.        ,  2.2706498 ,  1.52079393,  0.        ,  2.2706498 ,
             1.52079393,  1.        ,  0.        ],
           [ 2.2706498 ,  0.        ,  1.52079393,  2.2706498 ,  0.        ,
             1.52079393,  1.        ,  0.        ],
           [ 1.52079393,  1.52079393,  0.        ,  1.52079393,  1.52079393,
             0.        ,  1.        ,  0.        ],
           [ 0.        ,  2.2706498 ,  1.52079393,  0.        ,  2.2706498 ,
             1.52079393,  0.        ,  1.        ],
           [ 2.2706498 ,  0.        ,  1.52079393,  2.2706498 ,  0.        ,
             1.52079393,  0.        ,  1.        ],
           [ 1.52079393,  1.52079393,  0.        ,  1.52079393,  1.52079393,
             0.        ,  0.        ,  1.        ],
           [ 1.        ,  1.        ,  1.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  1.        ,  1.        ,
             1.        ,  0.        ,  0.        ]])
    """

    var_matrix_uu = block.calcBlockVariogramMatrix(controls, variogram)
    var_matrix_uv = block.calcBlockCoVariogramMatrix(controls, controls_co, crossvariogram)
    var_matrix_vu = block.calcBlockCoVariogramMatrix(controls_co, controls, crossvariogram)
    var_matrix_vv = block.calcBlockVariogramMatrix(controls_co, variogram_co)

    matrix_tempa = np.hstack((var_matrix_uu, var_matrix_uv))
    matrix_tempb = np.hstack((var_matrix_vu, var_matrix_vv))
    matrix_temp = np.vstack((matrix_tempa, matrix_tempb))

    horz_vec1a = [np.ones(controls.shape[0])]
    horz_vec1b = [np.zeros(controls_co.shape[0])]
    horz_vec1 = np.hstack((horz_vec1a, horz_vec1b))
        
    horz_vec2a = [np.zeros(controls.shape[0])]
    horz_vec2b = [np.ones(controls_co.shape[0])]
    horz_vec2 = np.hstack((horz_vec2a, horz_vec2b))

    matrix_temp = np.vstack((matrix_temp, horz_vec1, horz_vec2))

    vert_vec1 = np.transpose(horz_vec1)
    vert_vec1 = np.vstack((vert_vec1, [[0]], [[0]]))

    vert_vec2 = np.transpose(horz_vec2)
    vert_vec2 = np.vstack((vert_vec2, [[0]], [[0]]))

    co_matrix = np.hstack((matrix_temp, vert_vec1, vert_vec2))

    return co_matrix


def krigmatrix_rcok(controls, controls_co, 
                    variogram, variogram_co, crossvariogram):
    """ Calculates the Kriging-Matrix for Rescaled Co-Kriging.
    Note: Only for two variables.
     
     1...n_u        n_u+1...n_u+n_v     n_u+n_v+1 

    [Gamma_uu       Gamma_uv            1 ]
    [                                   . ] 
    [Gamma_uv       Gamma_vv            . ]
    [                                   1 ]
    [1 ................... 1            0 ]
     
    The Gammas are the (cross-)variogram matrices: 
    - Gamma_uu and Gamma_vv are the variograms of the two variables
    - Gamma_uv and Gamma_vu are the cross-variograms of the two variables (symmetric).
    One additional row/column for the constraint (one Lagrange multiplier).
    """

    var_matrix_uu = block.calcBlockVariogramMatrix(controls, variogram)
    var_matrix_uv = block.calcBlockCoVariogramMatrix(controls, controls_co, crossvariogram)
    var_matrix_vu = block.calcBlockCoVariogramMatrix(controls_co, controls, crossvariogram)
    var_matrix_vv = block.calcBlockVariogramMatrix(controls_co, variogram_co)

    matrix_tempa = np.hstack((var_matrix_uu, var_matrix_uv))
    matrix_tempb = np.hstack((var_matrix_vu, var_matrix_vv))
    matrix_temp = np.vstack((matrix_tempa, matrix_tempb))

    horz_vec1a = [np.ones(controls.shape[0])]
    horz_vec1b = [np.ones(controls_co.shape[0])]
    horz_vec1 = np.hstack((horz_vec1a, horz_vec1b))
        
    #horz_vec2a = [np.zeros(controls.shape[0])]
    #horz_vec2b = [np.ones(controls_co.shape[0])]
    #horz_vec2 = np.hstack((horz_vec2a, horz_vec2b))

    matrix_temp = np.vstack((matrix_temp, horz_vec1))

    vert_vec1 = np.transpose(horz_vec1)
    vert_vec1 = np.vstack((vert_vec1, [[0]]))

    #vert_vec2 = np.transpose(horz_vec2)
    #vert_vec2 = np.vstack((vert_vec2, [[0]], [[0]]))

    rco_matrix = np.hstack((matrix_temp, vert_vec1))

    return rco_matrix

    
def krigrhs_ok(controls, target, variogram):
    """ Calculates the right-hand-side of the Kriging system for Ordinary Kriging.
    >>> vario = variogram.ExponentialVariogram()
    >>> vario.setParameters({'range':2,'sill':3,})
    >>> controls = np.array([ [[1.,1.]],[[-1.,-1.]],[[0.5,-1.1]],[[-0.8,0.9]]])
    >>> target = np.array([0.,0.])
    >>> krigrhs_ok(controls, target, vario)
    array([ 1.52079393,  1.52079393,  1.36038741,  1.35698567,  1.        ])
    """
    # TODO: DANGER !!!
    # print controls.shape, target.shape
    if controls.shape[0]<2:
        controls = controls[0,:,:,:]

    nc = len(controls)
    rhs = np.ones(nc+1)

    # TODO: the newaxis in target might not be ideal (and necessary to change if block kriging included)
    for i in range(nc):
        rhs[i] = block.calcMeanVariogram(variogram, controls[i], target[np.newaxis, :])

    return rhs


def krigrhs_edk(controls, target, target_extdrift, variogram):
    """ Calculates the right-hand-side of the Kriging system for External Drift Kriging.
    >>> vario = variogram.ExponentialVariogram()
    >>> vario.setParameters({'range':2,'sill':3,})
    >>> controls = np.array([ [[1.,1.]],[[-1.,-1.]],[[0.5,-1.1]],[[-0.8,0.9]]])
    >>> target = np.array([0.,0.])
    >>> krigrhs_edk(controls, target, [6.], vario)
    array([ 1.52079393,  1.52079393,  1.36038741,  1.35698567,  1.        ,  6.        ])
    """
    if controls.shape[0]<2:
        controls = controls[0,:,:,:]
    rhs_ok = krigrhs_ok(controls, target, variogram)
    rhs = np.hstack((rhs_ok, target_extdrift[0]))

    return rhs


def krigrhs_cok(controls, controls_co, target, 
                variogram, variogram_co, crossvariogram):
    """ Calculates the right-hand-side of the Kriging system for Co-Kriging.

    [b_uu

    b_uv

    1
    0]

    The b-vectors are the vectors of auto-semivariances (gamma(u0,ui), gamma(u0,vj)).
    1 and 0 are added to honor the constraints.

    >>> vario = variogram.ExponentialVariogram()
    >>> vario.setParameters({'range':2,'sill':3,})
    >>> vario_co = variogram.ExponentialVariogram()
    >>> vario_co.setParameters({'range':2,'sill':3})
    >>> crossvario = variogram.ExponentialVariogram()
    >>> crossvario.setParameters({'range':2,'sill':3})
    >>> points = np.array([ [[1.,1.]],[[-1.,-1.]],[[0.,0.]] ])
    >>> points_co = np.array([ [[1.,1.]],[[-1.,-1.]],[[0.,0.]] ])
    >>> targ_point = np.array([1.,0.])
    >>> krigrhs_cok(points, points_co, targ_point, vario, vario_co, crossvario)
    array([ 1.18040802,  2.01923431,  1.18040802,  1.18040802,  2.01923431,
            1.18040802,  1.        ,  0.        ])
    """

    rhs = np.zeros(controls.shape[0] + controls_co.shape[0] + 2)
    for i in range(controls.shape[0]):
        rhs[i] = block.calcMeanVariogram(variogram, controls[i], target[np.newaxis, :])
    for j in range(controls.shape[0], controls.shape[0] + controls_co.shape[0]):
        #print(j)
        rhs[j] = block.calcMeanVariogram(crossvariogram, controls_co[j-controls.shape[0]], target[np.newaxis, :])
    rhs[controls.shape[0] + controls_co.shape[0]] = 1

    return rhs

def krigrhs_rcok(controls, controls_co, target, 
                 variogram, variogram_co, crossvariogram):
    """ Calculates the right-hand-side of the Kriging system for Rescaled Co-Kriging.

    [b_uu

    b_uv

    1]

    The b-vectors are the vectors of auto-semivariances (gamma(u0,ui), gamma(u0,vj)).
    1 is added to honor the constraint.
    """

    rhs = np.ones(controls.shape[0] + controls_co.shape[0] + 1)
    for i in range(controls.shape[0]):
        rhs[i] = block.calcMeanVariogram(variogram, controls[i], target[np.newaxis, :])
    for j in range(controls.shape[0], controls.shape[0] + controls_co.shape[0]):
        rhs[j] = block.calcMeanVariogram(crossvariogram, controls_co[j-controls.shape[0]], target[np.newaxis, :])
    #rhs[controls.shape[0] + controls_co.shape[0]] = 1

    return rhs
   

def krige_ok(controls, targets, controlvalues, variogram, n, n_type='Normal', verbose = True):
    """ 'Organizer function' for Ordinary Kriging.
    This function sets up the Kriging system and solves it for each given target
    >>> controls = np.array([ [[1.,1.]],[[-1.,-1.]],[[0.5,-1.1]],[[-0.8,0.9]]])
    >>> targets = np.array([[ 0., 0.]])
    >>> controlvalues = np.array([ 3.1,  4.,   2.,   5. ])
    >>> vario = variogram.ExponentialVariogram()
    >>> vario.setParameters({'range':2,'sill':3,})
    >>> n = controls.shape[0]
    >>> res = krige_ok(controls, targets, controlvalues, vario, n, n_type='Normal', verbose=True)
    >>> for i, result in enumerate(res):\
        print(result)
    (array([ 0.29404537,  0.29091373,  0.18633342,  0.22870749, -0.0024577 ]), 3.5063811706879497, 1.4235047111036763)
    """
    # preparation:
    #   include only the n closest points
    #   relative to point where estimation is carried out
    # print "start kdTree"
    k = min(n, len(controls))
    ctrlcentroids = block.calcCentroids(controls)           # centers (centroids) of the blocks
    controltree = scipy.spatial.cKDTree(ctrlcentroids)      # kdTree between all ctrlcentroids

    # up to here everything depends on measurements only (controls)
    # from here on: estimation for every individual point (target)
    # print "percent done: ",
    for target in targets:
        # preparation for closest points
        #qtree = controltree.query(target, k)     # find the k closest points in the controltree relative to the current target
        qtree = choose_points(target, controls, controltree, k, n_type)
        # tcontrols = controls[qtree[1][0],...]  # qtree[1][0] returns the indices of the k closest points in target
        tcontrols = controls[qtree[1]]           # tcontrols then contains the x,y coordinates of those points
        
        weight, interpol, estvar = solve_ok_equationSystem(tcontrols, controlvalues, target, variogram, qtree)

        yield weight, interpol, estvar

    del weight
    del interpol
    del estvar


def solve_ok_equationSystem(  tcontrols,
                              controlvalues,
                              target,
                              variogram,
                              qtree):
    # prepare kriging system
    ok_matrix = krigmatrix_ok(tcontrols, variogram)
    rhs = krigrhs_ok(tcontrols, target, variogram)

    # TODO clean this up!
    # solution for
    #    weights (lambdas),
    #    estimation (interpolated value)
    #    estimation variance at interpolated point (estvar)
    weight = np.linalg.solve(ok_matrix, rhs)

    # interpolated value = SUM(weight_i * measuredValue_i)
    # this is either
    #    np.sum(weight[:-1]*(controlvalues[qtree[1][0]].flatten()))
    # or
    # interpol = np.add.reduce(weight[:-1]*controlvalues[qtree[1][0]]).mean()
    interpol = (weight[:-1]*controlvalues[qtree[1]]).sum()

    # todo add term for block variance in estvar
    # estvar = np.add.reduce(weight[:-1]*rhs[:-1]) + weight[-1]
    # estvar = np.sum(weight[:-1]*rhs[:-1])  + weight[-1]  # (*) # Eq. 12.15 in Isaaks+


     # new attempt Claus 20140724

    # die lambdas (weights ohne mus)
    #lam = weight[:-1][np.newaxis, :]
    #print('weights')
    #print(lam)

    # variogram matrix der LHS
    #var_lhs = ok_matrix[:-1, :-1]

    # variogram array der rhs
    #var_rhs = rhs[:-1]

    #     print "var_lhs: ", var_lhs
    #     print "var_rhs: ", var_rhs

    # 1. summand der estimation variance
    # [n,n].sum() -- (n*n) Elemente
    #part_1 = (lam * lam.T * var_lhs).sum()

    # 2. summand der estimation variance
    # [1,n].sum() -- n Elemente
    #part_2 = (lam * var_rhs).sum()

    #estvar = (-1.0 * part_1) + (2.0 * part_2) # (*)


     # new attempt Sebastian 20162005
     # (results of the tagged (*) equations is the same)

    estvar = np.sum(rhs * weight)  # (*) # Eq. 12.16 in Isaaks (rewritten in terms of semivariances) or eq. 8.15 in Webster

    return weight, interpol, estvar


def krige_edk(controls,
              targets,
              controlvalues,
              control_extdrift,
              targets_extdrift,
              variogram,
              n,
              n_type='Normal',
              verbose=True,
              counters=[0, 0, 0]
              ):
    """ 'Organizer function' for External Drift Kriging.
    This function sets up the Kriging system and solves it for each given target
    #>>> vario = variogram.ExponentialVariogram()
    #>>> vario.setParameters({'range':2,'sill':3,})
    #>>> controls = np.array([ [[1.,1.]],[[-1.,-1.]],[[0.5,-1.1]],[[-0.8,0.9]]])
    #>>> targets = np.array([[[ 0., 0.]]])
    #>>> controlvalues = np.array([ 3.1,  4.,   2.,   5. ])
    #>>> control_extdrift = np.array([ 4.,  3.,  5.,  3.])
    #>>> targets_extdrift = np.array([ 6.,  3.])
    #>>> n = controls.shape[0]
    #>>> krige_edk(controls, targets, controlvalues, control_extdrift, targets_extdrift, vario, n, verbose = True)
    #edk matrix
    #[[ 0.          2.2706498   1.98055269  1.78198258  1.          4.        ]
    # [ 2.2706498   0.          1.58525759  1.84585059  1.          3.        ]
    # [ 1.98055269  1.58525759  0.          2.08978437  1.          5.        ]
    # [ 1.78198258  1.84585059  2.08978437  0.          1.          3.        ]
    # [ 1.          1.          1.          1.          0.          0.        ]
    # [ 4.          3.          5.          3.          0.          0.        ]]
    #weights
    #[ 0.35579101 -0.5169096   1.3221045  -0.1609859  -5.43188571  1.4486942 ]
    #interpolation result
    #0.874593200643

    """
    # temporarily fixed number of nearest controls until the final structure
    # can be put into the api

    # KDTree returns weird data, if more points are requested than are available
    # therefore limit the number if there are not enough points
    k = min(n, len(controls))
    # calculate the centroids of the blocks (if there are any). This should
    # be robust, if only points are given (as long as the points are passed
    # in via a 3-D (nblocks x npoints x dimension) array and npoints==1).
    ctrlcentroids = block.calcCentroids(controls)
    # set up the KDTree using the centroids
    controltree = scipy.spatial.cKDTree(ctrlcentroids)

    # bis hierher haengt alles nur von messwerten (controls) ab
    # ab jetzt: fuer jeden interpolationspunkt (target) interpolieren

    for target, target_extdrift in zip(targets, targets_extdrift):
        # query the controls-tree for the k nearest points to the target
        #qtree = controltree.query(target, k)
        qtree = choose_points(target, controls, controltree, k, n_type)
        # initialize temporary controls and external drift lists by
        # selecting the points returned by the tree query
        tcontrols = controls[qtree[1]]
        tcontrol_extdrift = control_extdrift[qtree[1]]

        # ----------------------------------------------------------------------
        # Check, if drift is
        # a) equal at all locations (no additional information)
        #    ==> OK
        # b) at measLoc outside range of drift at all other points (extrapolation)
        #    ==> OK
        # ----------------------------------------------------------------------

        if len(set(tcontrol_extdrift.flatten())) == 1:  # use Ordinary Kriging
            print("no ED")
            counters[0] += 1
            weight, interpol, estvar = solve_ok_equationSystem(tcontrols,
                                                              controlvalues,
                                                              target,
                                                              variogram,
                                                              qtree)

        elif np.logical_or((np.min(tcontrol_extdrift) > target_extdrift), (np.max(tcontrol_extdrift) < target_extdrift)):
            #print("ED outside range")
            counters[1] += 1
            weight, interpol, estvar = solve_ok_equationSystem(tcontrols,
                                                              controlvalues,
                                                              target,
                                                              variogram,
                                                              qtree)

        else:
           counters[2] += 1
           weight, interpol, estvar = solve_edk_equationSystem(tcontrols,
                                                               tcontrol_extdrift,
                                                               controlvalues,
                                                               target,
                                                               target_extdrift,
                                                               variogram,
                                                               qtree)

        yield weight, interpol, estvar, counters

    del weight
    del interpol
    del qtree
    del tcontrols
    del tcontrol_extdrift


def solve_edk_equationSystem(tcontrols,
                             tcontrol_extdrift,
                             controlvalues,
                             target,
                             target_extdrift,
                             variogram,
                             qtree):

    edk_matrix = krigmatrix_edk(tcontrols, tcontrol_extdrift, variogram)
    rhs = krigrhs_edk(tcontrols, target, target_extdrift, variogram)

    try:
        weight = np.linalg.solve(edk_matrix, rhs)
    except:
        # TODO: FIX THIS!
        print('Crazy Kriging Matrix --> Check Out Why!!!')
        raise Exception
        # print 'multiplying edk_matrix'
        rd_mat = np.ones_like(edk_matrix)
        rds = np.random.random(tcontrol_extdrift.shape[0]+2) * 0.05 + 0.95
        rd_mat[:, -1] = rds
        rd_mat[-1, :] = rds
        edk_matrix = edk_matrix * rd_mat
        weight = np.linalg.solve(edk_matrix, rhs)

    # interpol = np.add.reduce(weight[:-2]*controlvalues[qtree[1][0]]).mean()
    interpol = (weight[:-2]*controlvalues[qtree[1]]).sum()

    # TODO dangerous! Clean this up! Only via similarity
    # estvar = np.sum(weight[:-1]*rhs[:-1])  + weight[-1]

    # new attempt 20140724

    # die lambdas (weights ohne mus)
    lam = weight[:-2][np.newaxis, :]

    # variogram matrix der LHS
    var_lhs = edk_matrix[:-2, :-2]

    # variogram array der rhs
    var_rhs = rhs[:-2]

    # 1. summand der estimation variance
    # [n,n].sum() -- (n*n) Elemente
    part_1 = (lam * lam.T * var_lhs).sum()

    # 2. summand der estimation variance
    # [1,n].sum() -- n Elemente
    part_2 = (lam * var_rhs).sum()

    estvar = (-1.0 * part_1) + (2 * part_2)

    # new attempt Sebastian 20162005
     # (results of the tagged (*) equations is the same)

    estvar = np.sum(rhs * weight) # (*) # Eq. 12.16 in Isaaks (rewritten in terms of semivariances) or eq. 8.15 in Webster

    return weight, interpol, estvar


def krige_cok(controls, controls_co, 
              targets, 
              controlvalues, controlvalues_co, 
              variogram, variogram_co, crossvariogram,
              n, n_type='Normal', 
              verbose = True):
    """'Organizer function' for Co-Kriging.
    This function sets up the Co-Kriging system and solves it for each given target.
    """

    # preparation:
    #   include only the n closest points
    #   relative to point where estimation is carried out
    # print "start kdTree"
    # 1st: primary variable
    k = min(n, controls.shape[0])
    ctrlcentroids = block.calcCentroids(controls)
    controltree = scipy.spatial.cKDTree(ctrlcentroids)

    # 2nd: secondary variable
    k_co = min(n, controls_co.shape[0])
    ctrlcentroids_co = block.calcCentroids(controls_co)
    controltree_co = scipy.spatial.cKDTree(ctrlcentroids_co)

    # TODO: 
    # - new code for determining which points to choose for the estimation
    # - not just take the closest ones, but care for direction
    # e.g. : n closest ones from North-East, SE, NW, SW
    # - print histogram of distances of the chosen points 
    #  (should equal appr. the overall histogram)

    # up to here everything depends on measurements only (controls)
    # from here on: estimation for every individual point (target)
    # print "percent done: ",
    for target in targets:
        # preparation for closest points
        #print(target, k)
        #qtree = controltree.query(target, k)     # find the k closest points in the controltree relative to the current target
        qtree = choose_points(target, controls, controltree, k, n_type)
        # tcontrols = controls[qtree[1][0],...]  # qtree[1][0] returns the indices of the k closest points in target
        tcontrols = controls[qtree[1]]           # tcontrols then contains the x,y coordinates of those points

        # qtree_co = controltree_co.query(target, k_co)
        qtree_co = choose_points(target, controls_co, controltree_co, k_co, n_type)
        tcontrols_co = controls_co[qtree_co[1]]

        weight, interpol, estvar = solve_cok_equationSystem(tcontrols, tcontrols_co,
                                                            target,
                                                            controlvalues, controlvalues_co,
                                                            variogram, variogram_co, crossvariogram,
                                                            qtree, qtree_co)

        yield weight, interpol, estvar

    del weight
    del interpol
    del estvar


def solve_cok_equationSystem(tcontrols, tcontrols_co,  
                             target, 
                             controlvalues, controlvalues_co,
                             variogram, variogram_co, crossvariogram, 
                             qtree, qtree_co):
    """
    """
    # prepare kriging system

    cok_matrix = krigmatrix_cok(tcontrols, tcontrols_co, variogram, variogram_co, crossvariogram)
    rhs = krigrhs_cok(tcontrols, tcontrols_co, target, variogram, variogram_co, crossvariogram)
    # solution for
    #    weights (lambdas),
    #    estimation (interpolated value)
    #    estimation variance at interpolated point (estvar)
    
    try:
        weight = np.linalg.solve(cok_matrix, rhs)

    except:
        # TODO: FIX THIS?
        print('Crazy Kriging Matrix --> Check Out Why!!!')
        #np.savetxt(r'G:\sebastian_gnann\_code\sebastian_gnann_claus\store\cokrig\cok_matrix.txt',
        #           cok_matrix, fmt='%1.2f', delimiter=';', newline='\n')
        #np.savetxt(r'G:\sebastian_gnann\_code\sebastian_gnann_claus\store\cokrig\rhs.txt',
        #           rhs, fmt='%1.2f', delimiter=';', newline='\n')
        print(np.round(cok_matrix,2))
        print(np.round(rhs,2))
        raise Exception

    # interpolated value = SUM(weight_i * measuredValue_i)
    # this is either
    #    np.sum(weight[:-1]*(controlvalues[qtree[1][0]].flatten()))
    # or
    # interpol = np.add.reduce(weight[:-1]*controlvalues[qtree[1][0]]).mean()
    controlvalues_tot = np.hstack((controlvalues[qtree[1]], controlvalues_co[qtree_co[1]]))
    interpol = (weight[:-2]*controlvalues_tot).sum()
    estvar = np.sum(rhs * weight)

    return weight, interpol, estvar


def krige_rcok(controls, controls_co, 
              targets, 
              controlvalues, controlvalues_co, 
              variogram, variogram_co, crossvariogram,
              n, n_type='Normal',
              verbose = True):
    """Organizer function' for Rescaled Co-Kriging.
    This function sets up the Rescaled Co-Kriging system and solves it for each given target.

    """

    # preparation:
    #   include only the n closest points
    #   relative to point where estimation is carried out
    # print "start kdTree"
    # 1st: primary variable
    k = min(n, controls.shape[0])
    ctrlcentroids = block.calcCentroids(controls)
    controltree = scipy.spatial.cKDTree(ctrlcentroids)

    # 2nd: secondary variable
    k_co = min(n, controls_co.shape[0])
    ctrlcentroids_co = block.calcCentroids(controls_co)
    controltree_co = scipy.spatial.cKDTree(ctrlcentroids_co)

    # up to here everything depends on measurements only (controls)
    # from here on: estimation for every individual point (target)
    # print "percent done: ",
    for target in targets:
        # preparation for closest points
        #print(target, k)
        #qtree = controltree.query(target, k)     # find the k closest points in the controltree relative to the current target
        qtree = choose_points(target, controls, controltree, k, n_type) # use fct choose_points to get closest points depending on type
        # tcontrols = controls[qtree[1][0],...]  # qtree[1][0] returns the indices of the k closest points in target
        tcontrols = controls[qtree[1]]           # tcontrols then contains the x,y coordinates of those points

        # qtree_co = controltree_co.query(target, k_co)
        qtree_co = choose_points(target, controls_co, controltree_co, k_co, n_type)
        tcontrols_co = controls_co[qtree_co[1]]

        weight, interpol, estvar = solve_rcok_equationSystem(tcontrols, tcontrols_co,
                                                            target,
                                                            controlvalues, controlvalues_co,
                                                            variogram, variogram_co, crossvariogram,
                                                            qtree, qtree_co,
                                                            n)

        yield weight, interpol, estvar

    del weight
    del interpol
    del estvar


def solve_rcok_equationSystem(tcontrols, tcontrols_co,  
                             target, 
                             controlvalues, controlvalues_co,
                             variogram, variogram_co, crossvariogram, 
                             qtree, qtree_co,
                             n):
    """
    """
    # prepare kriging system
    rcok_matrix = krigmatrix_rcok(tcontrols, tcontrols_co, variogram, variogram_co, crossvariogram)
    rhs = krigrhs_rcok(tcontrols, tcontrols_co, target, variogram, variogram_co, crossvariogram)
    # TODO clean this up!
    # solution for
    #    weights (lambdas),
    #    estimation (interpolated value)
    #    estimation variance at interpolated point (estvar)
    
    try:
        weight = np.linalg.solve(rcok_matrix, rhs)
        
    except:
        # TODO: FIX THIS!
        print('Crazy Kriging Matrix --> Check Out Why!!!')
        #np.savetxt(r'G:\sebastian_gnann\_code\sebastian_gnann_claus\store\cokrig\rcok_matrix.txt',
        #           cok_matrix, fmt='%1.2f', delimiter=';', newline='\n')
        #np.savetxt(r'G:\sebastian_gnann\_code\sebastian_gnann_claus\store\cokrig\rrhs.txt',
        #           rhs, fmt='%1.2f', delimiter=';', newline='\n')
        print(np.round(rcok_matrix,2))
        print(np.round(rhs,2))
        raise Exception
    # interpolated value = SUM(weight_i * measuredValue_i)
    # this is either
    #    np.sum(weight[:-1]*(controlvalues[qtree[1][0]].flatten()))
    # or
    # interpol = np.add.reduce(weight[:-1]*controlvalues[qtree[1][0]]).mean()
    controlvalues_tot = np.hstack((controlvalues[qtree[1]], controlvalues_co[qtree_co[1]]))

    # here happens the "rescaling":
    # Z_est = sum(weights1*controls1) + sum(weights2*(controls1-mean2+mean1))
    # - the second variable is adjusted to have the same mean as the first one
    # - as means the arith. means of the control values is chosen
    # - to think of: global mean vs local mean?

    # adjust the control values of secondary variable
    # global mean
    adj_controlv_co = controlvalues_co[qtree_co[1]] - \
                      np.mean(controlvalues_co[:]) + \
                      np.mean(controlvalues[:])
    # local mean
    #adj_controlv_co = controlvalues_co[qtree_co[1]] - \
    #                  np.mean(controlvalues_co[qtree_co[1]]) + \
    #                  np.mean(controlvalues[qtree[1]])

    part1 = (weight[:n]*controlvalues[qtree[1]]).sum()
    part2 = (weight[n:-1]*adj_controlv_co).sum()
    interpol = part1 + part2

    estvar = np.sum(rhs * weight)

    return weight, interpol, estvar


def choose_points(target, controls, controltree, k, n_type):
    """ 
    Function for choosing "representative" points. 
    - Normal: n closest points independent of direction
    - Directional: n/4 closest points in each quarter (NE-SE-SW-NW)

    """

    #todo: check for high k values (k > n_tot / 5)
    #print('controls.shape[0]', controls.shape[0])
    
    if n_type == 'Directional':

        if k < controls.shape[0]/4: # if k is too large, i.e. too many points shall be used, 
                                    # the points are picked independent of direction

            limit = k/4 # limit, 1/4 of the control points are in this area.
                        # If n is not a multiple of 4, the n are chosen as follows: 
                        # e.g. n=10:
                        # ne: n=3, se n=3, sw n=2, nw n=2, 
                        # where the 3rd points in ne/se are closer than the 3rd points in sw/nw


            count_ne = 0 # counts how many points of this area are used
            count_se = 0
            count_sw = 0
            count_nw = 0
            count_tot = -1 # counts the overall points
            index_picked = [] # initial array with picked indices
            qtree = [] # initialize qtree array

            # The 4*k closest points are picked. It is iteratively checked in which area these points are lying.
            # The control point array is filled with points until k/4 points of each area are picked. 
            # If there are not enough points in each area after having checked the k*4 overall closest points,
            # the further points are picked without considering the direction.

            qtree_temp = controltree.query(target, k*5) # 5*k to ensure to have enough points.
            controls_temp = controls[qtree_temp[1]]

            for control_point in controls_temp:
                count_tot += 1
                            
                if control_point[0,0] > target[0] and control_point[0,1] > target[1] \
                    and count_ne < limit and (count_ne+count_se+count_sw+count_nw) < k:
                    # north-east
                    qtree.append(qtree_temp[1][count_tot])
                    count_ne += 1
                    index_picked.append(count_tot)

                if control_point[0,0] > target[0] and control_point[0,1] < target[1] \
                    and count_se < limit and (count_ne+count_se+count_sw+count_nw) < k:
                    # south-east
                    qtree.append(qtree_temp[1][count_tot])
                    count_se += 1
                    index_picked.append(count_tot)

                if control_point[0,0] < target[0] and control_point[0,1] < target[1] \
                    and count_sw < limit and (count_ne+count_se+count_sw+count_nw) < k:
                    # south-west
                    qtree.append(qtree_temp[1][count_tot])
                    count_sw += 1
                    index_picked.append(count_tot)

                if control_point[0,0] < target[0] and control_point[0,1] > target[1] \
                    and count_nw < limit and (count_ne+count_se+count_sw+count_nw) < k:
                    # north-west
                    qtree.append(qtree_temp[1][count_tot])
                    count_nw += 1
                    index_picked.append(count_tot)

                if (count_ne+count_se+count_sw+count_nw) == k:
                    break
            
            if len(qtree) < k:
                #print("Not enough points in each quarter... take some points independent of direction")
                #print('index picked', index_picked)
                #print('controls_qtree',controls_qtree)
                
                qtree_temp_left = np.delete(qtree_temp[1], index_picked, 0)

                for i in range(k-len(qtree)):
                    #print(i)
                    #print(qtree_temp_left[i])
                    qtree.append(qtree_temp_left[i])

            #print('len', len(qtree))
                   
            qtree = np.array([np.zeros_like(qtree), qtree]) # to imitate the original shape of the qtree array

        else: # if there are not enough overall points, the "normal" way is chosen
            # preparation for closest points
            print('Too few points, control points are chosen without considering direction')
            qtree = controltree.query(target, k)     # find the k closest points in the controltree relative to the current target


    elif n_type == 'Normal':
        # preparation for closest points
        qtree = controltree.query(target, k)     # find the k closest points in the controltree relative to the current target

    else:
        raise Exception('Wrong n_type, choose either Normal or Directional')

    #print(controls[qtree[1],0,:])
    #plt.scatter(controls[qtree[1],0,0], controls[qtree[1],0,1], marker='o')
    #plt.scatter(target[0], target[1], marker='o', color='red')
    #plt.show()

    #coord = np.array([controls[qtree[1],0,0], controls[qtree[1],0,1]])
    #H = ssd.pdist(coord.T, metric='euclidean')
    ##print(H)
    #hist, bins = np.histogram(H, bins=np.arange(0,np.round(np.max(H)/100)*100, 2000), normed=True)
    #width = 1. * (bins[1] - bins[0])
    #center = (bins[:-1] + bins[1:])/2
    #plt.bar(center, hist*100, align='center', width=width, facecolor = 'Grey')
    #plt.show()

    return qtree


def readVariogram(filename):
    """reads Variogram definitions and parameters from an ini-style configuration file
    """
    #from configparser import SafeConfigParser
    #import variogram

    parser = SafeConfigParser()
    parser.read(filename)

    vario = variogram.CompositeVariogram()

    for section in parser.sections():
        if section in variogram.__dict__:
            # the only way I found to create classes from their name
            variotmp = variogram.__dict__[section]()
            paramdict = {}

            for name, value in parser.items(section):
                paramdict[name]=float(value)

            variotmp.setParameters(paramdict)
            vario.addVariogram(variotmp)
        else:
            raise 'Variogram type not known: ' + section + \
                  '\nfirst, make sure that this isn''t due to a typo' + \
                  '\nthen check the available variogram types in variogram.py'

    return vario


def edk(  locmeasurements_xy
        , loctointerpolate_xy
        , measurements
        , exdriftmeasurements
        , exdriftatinterpolpts
        , compVar
        , weightfilename = 'Edk_weight_out.dat'
        , interpolfilename = 'Edk_out.dat'
        , estvarfilename = 'Edk_estvar_out.dat'
        , returnVals = False
        , n=10
        , counters=[0,0,0]):
    """ File I/O wrapper for External Drift Kriging.
    Reads files and prepares data for the External Drift Kriging organizer
    function
    """
    #------------------------------------------------------------
    #                                main routine for exdrkriging
    #------------------------------------------------------------ed
    # define names of input files
    #locmeasurements_xy      = "Edk_Input.dat"
    #loctointerpolate_xy     = "Target_Input.dat"
    #measurements            = "Measurements.dat"
    #exdriftmeasurements     = "Exdriftmeasurements.dat"
    #exdriftatinterpolpts    = "Exdriftatinterpolpts.dat"

    # read input files
    controls = locmeasurements_xy
    targets = loctointerpolate_xy
    controlvalues = measurements
    control_extdrift = exdriftmeasurements
    targets_extdrift = exdriftatinterpolpts

    variogram = compVar # readVariogram(variogramfile)

    # run external drift kriging

##    inputstring = '# input files:\n' \
##                   + '# controls:           ' + locmeasurements_xy  + '\n' \
##                   + '# targets:            ' + loctointerpolate_xy  + '\n' \
##                   + '# control values:     ' + measurements  + '\n' \
##                   + '# control ext drifts: ' + exdriftmeasurements  + '\n' \
##                   + '# target ext drifts:  ' + exdriftatinterpolpts  + '\n'


    # open output file
    interpolfile = open(interpolfilename, 'w')
    interpolfile.write( '# interpolation result file for external drift kriging\n')

    weightfile = open(weightfilename, 'w')
    weightfile.write( '# interpolation weights file for external drift kriging\n')

    estvarfile = open(estvarfilename, 'w')
    estvarfile.write( '# estimation variance file for ordinary kriging\n')
    for weights, interpol, estvar, counters in krige_edk(          controls,
                                                         targets,
                                                         controlvalues,
                                                         control_extdrift,
                                                         targets_extdrift,
                                                         variogram,
                                                         n,
                                                         verbose=False,
                                                         counters=counters):
        interpolfile.write('%6.3f\n' % (interpol))
        weightfile.write('\t'.join(['%6.4f' %(item) for item in weights]) + '\n')
        estvarfile.write('%10.5g \n' % (estvar))


    interpolfile.close()
    weightfile.close()
    estvarfile.close()

    if returnVals == True:
        return  weights, interpol, estvar, counters
    else:
        pass

    print("done")
    del weights
    del interpol


def ok(locmeasurements_xy, loctointerpolate_xy, measurements, variogram
       , n
       , weightfilename = 'Ok_weight_out.dat'
       , interpolfilename = 'Ok_out.dat'
       , estvarfilename = "Ok_estvar_out.dat"
       , returnVals = False):
    """ File I/O wrapper for Ordinary Kriging.
    Reads files and prepares data for the Ordinary Kriging organizer function
    """
    #------------------------------------------------------------
    #                                         main routine for OK
    #------------------------------------------------------------
    #print "Executing Ordinary Kriging Program \n"
    # define names of input files
    #locmeasurements_xy      = "Ok_Input.dat"
    #loctointerpolate_xy     = "Target_Input.dat"
    #measurements            = "Measurements.dat"

    # read input files
    #print "... reading input files"
    controls = locmeasurements_xy # readBlockCoordinates(locmeasurements_xy)
    targets = loctointerpolate_xy #readBlockCoordinates(loctointerpolate_xy)
    controlvalues = measurements #read_file(measurements)

    variogram = variogram #readVariogram(variogramfile)

    #print "executing kriging"

##    inputstring = '# input files:\n' \
##                   + '# controls:           ' + locmeasurements_xy  + '\n' \
##                   + '# targets:            ' + loctointerpolate_xy  + '\n' \
##                   + '# control values:     ' + measurements  + '\n'

    # open output file
    interpolfile = open(interpolfilename, 'w')
    interpolfile.write( '# interpolation result file for ordinary kriging\n')
    #interpolfile.write( inputstring )

    weightfile = open(weightfilename, 'w')
    weightfile.write( '# interpolation weights file for ordinary kriging\n')
    #weightfile.write( inputstring )

    estvarfile = open(estvarfilename, 'w')
    estvarfile.write( '# estimation variance file for ordinary kriging\n')
    #estvarfile.write( inputstring )

    for weights, interpol, estvar in krige_ok(controls, targets, controlvalues
                                      , variogram, n, verbose=False):
        #if returnVals == False:
        interpolfile.write('%10.5g \n' % (interpol))
        weightfile.write('\t'.join(['%6.6f' %(item) for item in weights]) + '\n')
        estvarfile.write('%10.5g \n' % (estvar))

    interpolfile.close()
    weightfile.close()
    estvarfile.close()

    if returnVals == True:
        return weights, interpol, estvar
    else:
        pass
    print("done")


def read_file(filename):
    """ Simple file input function.
    Reads lines from a file, separates columns by whitespace and returns
    the results as a one- or two-dimensional numpy ndarray.
    """
    f = open(filename, 'r')
    outarray = []
    for line in f.readlines():
        split = line.strip().split()
        outarray.append([ float(item) for item in split ])
    f.close()
    return np.array(outarray).squeeze()


def _test():
    import doctest
    doctest.testmod()


if __name__ == "__main__":
    print('this is a pure module file')
    print('running doctests on module')
    _test()
    print('doctests finished')
