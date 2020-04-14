#!/usr/bin/env python

import numpy as np
import scipy.special
import pylab as plt



def main():

    h = np.linspace(0,2.5,25)



    c1 = type_matern( h,v=1.0)

    c2 = type_hole(h)

    c3 = type_exp(h)

    c4 = type_gau(h)

    c5 = type_sph(h)

    c6 = type_lin(h)





    plt.figure()

    plt.plot(h, c3, '--', color='black', label='exponential')

    plt.plot(h, c4, '-.', color='black', label='gaussian')

    plt.plot(h, c5, '-', color='black', label='spherical')

    plt.plot(h, c1, '.-',  color='black', label='matern')

    plt.plot(h, c2, ':',  color='black', label='hole effect')

    plt.legend()

    plt.xlabel('h')

    plt.ylabel('R')

    plt.ylim(-0.4,1)

    plt.tight_layout()

    #plt.savefig('plt.png',dpi=300)

    plt.show()



def Covariogram_return_func(model='1.0 Exp(1.0)', ):

    '''

    returns the function for later assignment of h!

    '''

    covfun = lambda h: Nested_Cov(h, model=model)

    return covfun



##def Covariogram(h, model='1.0 Exp(1.0)'):
##
##    C = Nested_Cov(h, model=model)
##
##    return C



##def Correlogram(h, model='1.0 Exp(1.0)'):

##    C = Nested_Cov(h, model=model)

##    Rho = C / C[0]

##    return Rho



def Variogram(h, model='1.0 Exp(1.0)'):

    C = Nested_Cov(h, model=model)

    gamma = C[0] - C

    return gamma



def Nested_Cov(h, model='1.0 Exp(1.0)'):

    '''

    h... distance vector

    model...gstat like string

        *possible models:

            Hol = Hole-effect (Exponential times cosinus)

            Mat = Matern

            Exp = Exponential

            Sph = Spherical

            Gau = Gaussian

            Lin = Linear

            Nug = Nugget

            Pow = Power-law

            Cau = Cauchy

            e.g.: '1.0 Exp(3.7) + 1.9 Mat(2.2)^0.5 + 0.3 Nug(666)'

        *the matern and hole model require an additional parameter:

            'sill Mat(range)^parameter'

        *the nugget model requires a range also, but it is not taken into account!

            'sill Nug(0)'

        *every other model:

            'sill Typ(range)''

        *superposition is possiblewith ' + '

    '''

    h = np.atleast_1d(np.array(h).astype(float))



    # check models

    models = model.split('+')

    models = np.array(models)



    # go through models:

    C = np.zeros(h.shape)

    for submodel in models:

        submodel = submodel.strip()

        Sill  = submodel.split('(')[0].strip()[:-3].strip()

        Range = submodel.split('(')[1].split(')')[0]

        Type  = submodel.split('(')[0].strip()[-3:]



        Sill = np.array(Sill).astype('float')

        if Sill <= 0:

            Sill = np.array((0.0))



        Range = np.abs(np.array(Range).astype('float'))

        if Range <= 0:

            Range = np.array((0.0))



        Type = np.array(Type)



        # calculate covariance:

        if Type == 'Mat':

            Param = submodel.split('^')[1].strip()

            Param = np.array(Param).astype('float')

            c0 = C[np.where(h==0)]

            C += type_matern(h, v=Param, Range=Range, Sill=Sill)

            C[np.where(h==0)] = c0 + Sill

        elif Type == 'Hol':

            C += type_hole(h, Range=Range, Sill=Sill)

        elif Type == 'Exp':

            C += type_exp(h, Range=Range, Sill=Sill)

        elif Type == 'Sph':

            C += type_sph(h, Range=Range, Sill=Sill)

        elif Type == 'Gau':

            C += type_gau(h, Range=Range, Sill=Sill)

        elif Type == 'Lin':

            C += type_lin(h, Range=Range, Sill=Sill)

        elif Type == 'Nug':

            C[np.where(h==0)] += Sill

        elif Type == 'Pow':

            c0 = C[np.where(h==0)]

            C += type_power(h, Range=Range, Sill=Sill)

            C[np.where(h==0)] = c0 + Sill

            print ('Not sure if it works yet!')

        elif Type == 'Cau':

            alpha = submodel.split('^')[1].strip()

            alpha = np.array(alpha).astype('float')

            beta = submodel.split('^')[2].strip()

            beta = np.array(beta).astype('float')

            C += type_cauchy(   h,

                                Range=Range,

                                Sill=Sill,

                                alpha=alpha,

                                beta=beta)



    return C



def type_hole(h, Range=1.0, Sill=1.0):

    h = np.array(h)

    C = np.ones(h.shape)*Sill

    ix = np.where(h>0)

    C[ix] = Sill*(np.sin(np.pi*h[ix]/Range)/(np.pi*h[ix]/Range))

    return C



def type_exp(h, Range=1.0, Sill=1.0):

    h = np.array(h)

    return Sill * (np.exp(-h/Range))



def type_sph(h, Range=1.0, Sill=1.0):

    h = np.array(h)

    return np.where(h>Range, 0,

            Sill * (1 - 1.5*h/Range + h**3/(2*Range**3)))



def type_gau(h, Range=1.0, Sill=1.0):

    h = np.array(h)

    return Sill * np.exp(-h**2/Range**2)



def type_lin(h, Range=1.0, Sill=1.0):

    h = np.array(h)

    return np.where(h>Range, 0, Sill * (-h/Range + 1))



def type_matern(h, v=0.5, Range=1.0, Sill=1.0):

    '''

    Matern Covariance Function Family:

        v = 0.5 --> Exponential Model

        v = inf --> Gaussian Model

    '''

    h = np.array(h)



    # for v > 100 shit happens --> use Gaussian model

    if v > 100:

        c = type_gau(h, Range=1.0, Sill=1.0)

    else:

        Kv  = scipy.special.kv      # modified bessel function of second kind of order v

        Tau = scipy.special.gamma  # Gamma function



        fac1 = h / Range * 2.0*np.sqrt(v)

        fac2 = (Tau(v)*2.0**(v-1.0))

        c = Sill * 1.0 / fac2 * fac1**v * Kv(v, fac1)



        # set nan-values at h=0 to sill

        c[np.where(h==0)]=Sill



    return c



##def type_matern1(h, v=0.5, Range=1.0, Sill=1.0):

##    '''

##    Matern Covariance Function Family:

##        v = 0.5 --> Exponential Model

##        v = inf --> Gaussian Model

##    '''

##    h = np.array(h)

##

##    # for v > 100 shit happens --> use Gaussian model

##    if v > 100:

##        c = type_gau(h, Range=1.0, Sill=1.0)

##    else:

##        Kv = scipy.special.kv      # modified bessel function of second kind of order v

##        Tau = scipy.special.gamma  # Gamma function

##

##        c = Sill * (2**(v-1)*Tau(v))**-1 * (h/Range)**v * Kv(v,h/Range)

##

##        # set nan-values at h=0 to sill

##        c[np.where(h==0)]=Sill

##

##    return c



def type_power(h, Range=1.0, Sill=1.0):

    h = np.array(h)

    return Sill - h**Range



def type_cauchy(h, Range=1., Sill=1., alpha=1., beta=1.0):

    """

    alpha >0 & <=2 ... shape parameter

    beta >0 ... parameterizes long term memory

    """

    h = np.array(h).astype('float')

    return Sill*(1 + (h/Range)**alpha)**(-beta/alpha)





def find_maximum_range(model='1.0 Exp(1.0)', rho_thresh=0.01):

    '''

    returns range of the model where correlation is rho_thresh

    '''

    # check models

    models = model.split('+')

    models = np.array(models)

    # go through models:

    maxrange = 0

    for submodel in models:

        submodel = submodel.strip()

        Range = submodel.split('(')[1].split(')')[0]

        Range = float(Range)

        if maxrange<Range:

            maxrange = Range



    # search integralscale...

    integralscale = 0

    correlation = Nested_Cov(integralscale, model=model)

    while correlation > rho_thresh:

        integralscale += maxrange/10.0

        correlation = Nested_Cov(integralscale, model=model)

    # maybe insert something like ifnan ifinf ifzero...then set to maxrange*5

    integralscale = max(maxrange*3,   integralscale)

    integralscale = min(maxrange*100, integralscale)

    return integralscale







if __name__ == '__main__':

    main()

