#!/usr/bin/env python
import numpy as np

def ExponentialVariogram(var_sill,var_range, dist):
    return var_sill * (1 - np.exp(-(dist/var_range)))

def SphericalVariogram(var_sill,var_range, dist):
    return np.where(dist < var_range, var_sill * (1.5*(dist/var_range) - 0.5*(dist**3/var_range**3)), var_sill)

def GaussianVariogram(var_sill,var_range, dist):
    return var_sill * (1 - np.exp(-((dist**2)/(var_range)**2)))

def NuggetVariogram(var_sill,var_range, dist):
    return np.where(dist > 0., (sill), 0.)

def LinearVariogram(var_sill,var_range, dist):
    return abs(dist)
    
# def MaternVariogram(var_sill,var_range):
#     v = 0.5
#     if v > 100:
#             c = GaussianVariogram()
#             params = {'sill':1., 'range':1.}
#             c.setParameters(params)
# ##            c.calculate(dst)
#     else:
#             Kv = sp.special.kv      # modified bessel function of second kind of order v
#             Tau = sp.special.gamma  # Gamma function

#             fac1 = dist/var_range * 2.0*np.sqrt(v)
#             fac2 = (Tau(v)*2.0**(v-1.0))
#             c = var_sill * 1.0 / fac2 * fac1**v * Kv(v, fac1)

#             # set nan-values at h=0 to sill
#             c[np.where(dst==0)] = var_sill

#     return (1.0 - c) * variance

# def CompositeVariogram(var_sill,var_range):
#     def __init__(self, variogramList=None):
#         self.variogramList = []
#         if variogramList is not None:
#             for item in variogramList:
#                 self.addVariogram(item)


#     def addVariogram(self, variogram):
#         self.variogramList.append(variogram)


#     def calculate(self, distance):
#         """
#         >>> import numpy as np
#         >>> nug = NuggetVariogram({'sill':3.})
#         >>> exp = ExponentialVariogram({'sill':4.,'range':5.,})
#         >>> cv = CompositeVariogram()
#         >>> cv.addVariogram(nug)
#         >>> cv.addVariogram(exp)
#         >>> cv.calculate(np.array([[1.,2.],[3.,4.]]))
#         array([[ 3.72507699,  4.31871982],
#                [ 4.80475346,  5.20268414]])
#         """
#         result = np.zeros(distance.shape)
#         for vario in self.variogramList:
#             result = result + vario.calculate(distance)
#         return result