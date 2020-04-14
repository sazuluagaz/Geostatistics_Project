#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import numpy as np
__author__ = "Claus Haslauer (claus.haslauer@iws.uni-stuttgart.de)"
__version__ = "$Revision: 1.0 $"
__date__ = datetime.date(2020, 2, 14)
__copyright__ = "Copyright (c) 2017 Claus Haslauer"
__license__ = "Python"


import os
try:
    user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
    print("PYTHONPATHs: ")
    for cur_ix, cur_path in enumerate(user_paths): print("  ", cur_ix, ": ", cur_path.strip('\r\n'))
    print()
except KeyError:
    user_paths = []
    

import scipy.stats as st
import scipy.spatial as sp
import logging

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:
    logging.warn('error while importing matplotlib')
    
import spatial.correlationfunctions.th_cov_func as covfun
#import phy

def main():
    srf = cholesky_random_field(
                    domainsize=(50,50),
                    covmod = '1.0 Mat(20)^0.5'
                    )
    srf.plt2D()


##    crf = cholesky_random_field(
##                    domainsize=(10,10),
##                    covmod = '1.0 Mat(3)^1.5',
##                    periodic=True
##                                )
##    a=42


    # Spek = []
    # for rainsch in [2]: # ,5,10,20]:
    #     srf = cholesky_random_field(
    #                     domainsize=(100,100),
    #                     covmod = '1.0 Mat(%i)^1.5'%rainsch
    #                     )
    #
    #     nmc = 100
    #     spektrum = []
    #     for i in range(nmc):
    #         print(i, end=' ')
    #         field = srf.simnew()
    #         f2 = field#**2
    #         spektrum.append(np.abs(np.fft.fftn(f2))[:,0])
    #     spektrum = np.array(spektrum)
    #     Spek.append(spektrum.mean(axis=0))
    #
    # plt.figure()
    # for i in range(len(Spek)):
    #     plt.plot(np.arange(1,101), Spek[i], label='%i'%i)
    # plt.legend()
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()


class cholesky_random_field(object):
    def __init__(self,
                 domainsize = (25,25),
                 covmod     = '1.0 Mat(2)^1',
                 periodic   = False,
                 ):
        self.domainsize = domainsize
        self.covmod = covmod

        self.ncoords = np.prod(domainsize)
        self.ndim = len(domainsize)

        self.grid = np.mgrid[[slice(0,n,1) for n in domainsize]]
        self.xy = ((self.grid.flatten()).reshape(self.ndim,self.ncoords)).T

        #calculate distance matrix
        D = sp.distance_matrix(self.xy,self.xy)
        if periodic == True: # set some distances to zero!
            print("Not Implemented")
            raise Exception

        #calculate covariance matrix (exponential variogram model)
        self.Q = covfun.Covariogram(D,covmod)
        #LU decomposition
        self.L = np.linalg.cholesky(self.Q)

        self.Y = self.simnew()

    def getY(self):
        return self.Y

    def simnew(self):
        #draw independent normals
        y = np.random.normal(0,1,self.ncoords)
        #dot product with independent normal
        self.Y = (np.dot(self.L, y)).reshape(self.domainsize)
        return self.Y

    def plt2D(self):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
        im3 = ax[0].imshow(self.Y, 
                           interpolation='nearest',
                           extent = [0.5,self.domainsize[0]+0.5, 0.5,self.domainsize[1]+0.5],
                           origin='lower')
        ax[0].set_xlabel('x coordinate')
        ax[0].set_ylabel('y coordinate')

        divider3 = make_axes_locatable(ax[0])
        cax3 = divider3.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im3, cax=cax3, extend='both', label='$\mathcal{N}(0,1)$')

        #print(self.Y.flatten().shape)
        ax[1].hist(self.Y.flatten(), 20, color='blue', density=True, label='simulated')
        
        x = np.linspace(st.norm.ppf(0.01), st.norm.ppf(0.99), 100)
        ax[1].plot(x, st.norm.pdf(x), 'r--', lw=1, alpha=0.6, label='theoretical')
        
        ax[1].set_xlabel('$\mathcal{N}(0,1)$')
        ax[1].legend()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
