import numpy as np
import scipy.special as sps
import scipy as sp

"""
a python class to describe spatial dependence using variograms
Thomas Pfaff & Claus Haslauer
2015
claus.haslauer@uni-tuebingen.de
"""

class Variogram():
    def __init__(self, parameters=None):
        self.parameters = parameters

    def calculate(self, distance):
        """
        >>> vario = Variogram()
        >>> vario.calculate(3)
        """
        self.checkParameters()


    def setParameters(self, parameters):
        self.parameters = parameters


    def getParameters(self):
        return self.parameters


    def checkParameters(self):
        pass


class ExponentialVariogram(Variogram):
    def checkParameters(self):
        keys = ['sill', 'range']
        if any([key not in self.parameters for key in keys] ):
            raise Exception('Not all parameters needed for variogram were given')


    def calculate(self, distance):
        """
        >>> vario = ExponentialVariogram()
        >>> params = {'sill':3., 'range':4.}
        >>> vario.setParameters(params)
        >>> vario.calculate(3.)
        1.58290034178
        >>> falseparams = {'range':4.}
        >>> vario.setParameters(falseparams)
        >>> vario.calculate(3.)
        Traceback (most recent call last):
          ...
        Exception: Not all parameters needed for variogram were given
        """
        #Exponential Variogram needs
        # sill
        # range
        Variogram.calculate(self, distance)
        return self.parameters['sill'] * (1 - np.exp(-distance/self.parameters['range']))


class SphericalVariogram(Variogram):
    def calculate(self, distance):
        """
        >>> vario = SphericalVariogram()
        >>> params = {'sill':3., 'range':4.}
        >>> vario.setParameters(params)
        >>> vario.calculate(3.)
        array(2.7421875)
        """
        #Spherical Variogram needs
        # sill
        # range
        Variogram.calculate(self, distance)
        # CHANGED CPH: ich denke so ist Spherical Variogram besser
        # alte version:
        #     return self.parameters['sill'] * (1 - (-distance/self.parameters['range']))
        return np.where(distance < self.parameters['range'],
                           self.parameters['sill'] * (1.5*(distance/self.parameters['range']) - 0.5*(distance**3/self.parameters['range']**3)),
                           self.parameters['sill'])


class GaussianVariogram(Variogram):
    def calculate(self, distance):
        """
        >>> vario = GaussianVariogram()
        >>> params = {'sill':3., 'range':4.}
        >>> vario.setParameters(params)
        >>> vario.calculate(3.)
        1.29065152581
        """
        #Sperical Variogram needs
        # sill
        # range
        Variogram.calculate(self, distance)
        # CHANGED CPH: ich denke so ist Spherical Variogram besser
        # alte version:
        #     return self.parameters['sill'] * (1 - (-distance/self.parameters['range']))
        return self.parameters['sill'] * (1 - np.exp(-((distance**2)/(self.parameters['range'])**2)))


class NuggetVariogram(Variogram):
    def calculate(self, distance):
        """
        >>> vario = NuggetVariogram()
        >>> params = {'sill':3.}
        >>> vario.setParameters(params)
        >>> vario.calculate([4.])
        array([3.0])
        """
        # Nugget Variogram needs
        # sill
        # This value is returned for all distances except 0. There, even a
        # nugget variogram has to be 0
        dst = np.asanyarray(distance)
        Variogram.calculate(self, dst)

        sill = self.parameters['sill']
        variance = self.parameters['variance']

        return np.where(dst > 0., (sill*variance), 0.)
        #return np.where(dst > 0., (sill), 0.)


class CompositeVariogram():
    def __init__(self, variogramList=None):
        self.variogramList = []
        if variogramList is not None:
            for item in variogramList:
                self.addVariogram(item)


    def addVariogram(self, variogram):
        self.variogramList.append(variogram)


    def calculate(self, distance):
        """
        >>> import numpy as np
        >>> nug = NuggetVariogram({'sill':3.})
        >>> exp = ExponentialVariogram({'sill':4.,'range':5.,})
        >>> cv = CompositeVariogram()
        >>> cv.addVariogram(nug)
        >>> cv.addVariogram(exp)
        >>> cv.calculate(np.array([[1.,2.],[3.,4.]]))
        array([[ 3.72507699,  4.31871982],
               [ 4.80475346,  5.20268414]])
        """
        result = np.zeros(distance.shape)
        for vario in self.variogramList:
            result = result + vario.calculate(distance)
        return result


class LinearVariogram(Variogram):
    def calculate(self, distance):
        """
        >>> vario = LinearVariogram()
        >>> params = {'slope':1.}
        >>> vario.setParameters(params)
        >>> vario.calculate([2.])
        array([ 2.0])
        """
        # Linear Variogram only takes the slope
        # sill
        # This value is returned for all distances except 0. There, even a
        # nugget variogram has to be 0
        dst = np.asanyarray(distance)
        Variogram.calculate(self, dst)

        return self.parameters['slope']*dst

# class MaternVariogram(Variogram):
#     def calculate(self, distance):
#         """
#         >>> m = MaternVariogram({'range':5., 'sill':1.,'shape':3.0})
#         >>> m.calculate([2.0])
#         array([ 0.01962106])
#         """
#         dst = np.asanyarray(distance)
#         Variogram.calculate(self, dst)
#
#         sill = self.parameters['sill']
#         rng = self.parameters['range']
#         shape = self.parameters['shape']
#         variance = self.parameters['variance']
#
#         hnorm = dst/rng
#         fac = 2.0*np.sqrt(shape)
#         Kv = sps.kv(shape,(hnorm*fac))
#
#         # vario = sill * (1 - ((2**(1.-shape))/(sps.gamma(shape)))*((hnorm*fac)**shape)*Kv)
#         vario = sill * ( ((2**(1.-shape))/(sps.gamma(shape)))*((hnorm*fac)**shape)*Kv)
#         vario[np.where(dst==0)] = sill
#
#         return (1.0 - vario) * variance
#         #return vario * variance

class MaternVariogram(Variogram):
    '''
    Matern Covariance Function Family:
        v = 0.5 --> Exponential Model
        v = inf --> Gaussian Model
    def type_matern(h, v=0.5, Range=1.0, Sill=1.0):
    '''
    def calculate(self, distance):
        dst = np.asanyarray(distance)
        Variogram.calculate(self, dst)

        sill = self.parameters['sill']
        rng = self.parameters['range']
        v = self.parameters['shape']
        variance = self.parameters['variance']

        # for v > 100 shit happens --> use Gaussian model
        if v > 100:
            c = GaussianVariogram()
            params = {'sill':1., 'range':1.}
            c.setParameters(params)
##            c.calculate(dst)
        else:
            Kv = sp.special.kv      # modified bessel function of second kind of order v
            Tau = sp.special.gamma  # Gamma function

            fac1 = dst / rng * 2.0*np.sqrt(v)
            fac2 = (Tau(v)*2.0**(v-1.0))
            c = sill * 1.0 / fac2 * fac1**v * Kv(v, fac1)

            # set nan-values at h=0 to sill
            c[np.where(dst==0)] = sill

        return (1.0 - c) * variance

def _test():
    import doctest
    print('running doctests on module')
    doctest.testmod()
    print('doctest finished')


if __name__ == '__main__':
    _test()
