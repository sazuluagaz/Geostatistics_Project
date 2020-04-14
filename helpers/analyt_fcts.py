import numpy as np

# --------------------------------------------------------------------------------
#                                                                      exponential
# --------------------------------------------------------------------------------
def exp2_cdf(parameters, x):
    cdf = 1.0 - np.exp(-parameters[0] * (x - parameters[1]))
    return cdf

def exp2_ppf(parameters, zz):
    #   CPH: so ein Docstring fehlt zu jeder Funktion... Du wirst es mir spaeter danken...
    """
    Inverse of a two-parametric exponential distribution function
    y = (ln(1-x)/-lambda)+y0

    """
    # so wir ein Fehler ausgerufen, wenn Du aus vershehen
    # die falsche Anzahl an Parametern uebergibst.
    # Das ist extrem nuetzlich um Fehler zu vermeiden!
    if len(parameters) != 2:
        raise TypeError("A two parametric distribution is allowed only two parameters!")
    lamb = parameters[0]
    y0 = parameters[1]
    out = (np.log(1-zz)/-lamb)+y0
    return out


def exp2_pdf(parameters, x):
    pdf = parameters[0] * np.exp(- parameters[0] * (x-parameters[1]))
    return pdf


def exp2_E(opt):
    """
    Rechnet Erwartungswert aus
    x0 + (1/lambda)
    """
    expectation = opt[1] + (1.0 / opt[0])
    return expectation


def exp2_V(opt):
    var = 1.0 / (opt[0]**2)
    return var
    
    

    
# --------------------------------------------------------------------------------
#                                                                          Weibull
# --------------------------------------------------------------------------------

def weibull2_ab_cdf(parameters, x):
    alpha = parameters[0]
    beta = parameters[1]
    cdf = 1.0 - np.exp(-alpha * x**beta)
    return cdf


def weibull2_ab_ppf(parameters, zz):
    #   CPH: so ein Docstring fehlt zu jeder Funktion... Du wirst es mir spaeter danken...
    """
    Inverse of a two-parametric Weibull distribution function
    y = (ln(1-x)/-alpha)**(1/beta)

    """
    # so wir ein Fehler ausgerufen, wenn Du aus vershehen
    # die falsche Anzahl an Parametern uebergibst.
    # Das ist extrem nuetzlich um Fehler zu vermeiden!
    if len(parameters) != 2:
        raise TypeError("A two parametric distribution is allowed only two parameters!")
    alpha = parameters[0]
    beta = parameters[1]
    out = (np.log(1-zz)/(-alpha))**(1/beta)
    return out


def weibull2_ab_pdf(parameters, x):
    alpha = parameters[0]
    beta = parameters[1]
    pdf = alpha * beta * x**(beta - 1.) * np.exp(-alpha * x**beta)
    return pdf


def weibull2_ab_E(opt):
    """
    Rechnet Erwartungswert aus
    alpha**(1/beta)*gamma(1/beta +1)

    opt[0][0] ... alpha
    opt[0][1] ... beta
    """
    alpha = opt[0]
    beta = opt[1]
    part1 = alpha**(-1.0/beta)
    # part2 = np.exp(special.gammaln( (1.0/beta)+1.0 ))
    part2 = ssp.gamma( (1.0/beta)+1.0 )
    # print "part1: %f | part2: %f" % (part1, part2)
    expectation = part1 * part2
    return expectation


def weibull2_ab_V(opt):
    a = float(opt[0])
    b = float(opt[1])
    var =(a**(-2/b)) * (ssp.gamma(2/b + 1)-(ssp.gamma(1/b+1))**2)
    return var

# --------------------------------------------------------------------------------
#                                                                          Gubmebl
# --------------------------------------------------------------------------------


def gumbel2_cdf(parameters, x):
    beta = parameters[1]
    mu = parameters[0]
    z = (x-mu)/beta
    cdf = np.exp(- np.exp(-z) )
    return cdf

def gumbel2_ppf(parameters, zz):
    #   CPH: so ein Docstring fehlt zu jeder Funktion... Du wirst es mir spaeter danken...
    """
    Inverse of a two-parametric Gumbel distribution function
    y = (-beta*ln(-ln(x)))+mu

    """
    # so wir ein Fehler ausgerufen, wenn Du aus vershehen
    # die falsche Anzahl an Parametern uebergibst.
    # Das ist extrem nuetzlich um Fehler zu vermeiden!
    if len(parameters) != 2:
        raise TypeError("A two parametric distribution is allowed only two parameters!")
    beta = parameters[1]
    mu = parameters[0]

    # out = -1 * ((beta*np.log(- np.log(zz)))) + mu
    out = mu - (beta*(np.log(- np.log(zz))))
    return out


def gumbel2_pdf(parameters, x):
    beta = float(parameters[1])
    mu = float(parameters[0])
    z = (x-mu)/beta
    pdf = (1.0/beta) * np.exp(-(z + np.exp(-z)))
    return pdf

def gumbel2_E(opt):
    """
    Rechnet Erwartungswert aus
    mu + gamma * beta
    gamma ist die Euler Masceroni Konstante
    """
    beta = opt[1]
    mu = opt[0]
    expectation = mu + beta*0.5772156649015328606
    return expectation

def gumbel2_V(opt):
    beta = opt[1]
    mu = opt[0]
    var = ((np.pi**2)/6.) * beta**2.
    return var
