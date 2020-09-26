import numpy as np
from numba import njit

MIN_SIGMA_DEFAULT = 1.0e-100

GAUSS_DTYPE = [
    ('weight', 'f8'),
    ('mean', 'f8'),
    ('sigma', 'f8'),
    ('icovar', 'f8'),
    ('norm', 'f8'),

    # not used in processing but useful later
]


def run_em(
    *,
    data,
    guess,
    maxiter,
    tol=1.0e-6,
    **constraint_kw
):
    """
    Parameters
    ----------
    x: array
        data to fit
    guess: gmix
        Starting gaussian mixture
    maxiter: int
        Max number of iterations
    tol: float, optional
        Consider converged when log likelihood changes
        by less than this amount

    weight_low: scalar or array
        Lower bound on the amplitude for one or all of the gaussians
    weight_high: scalar or array
        Upper bound on the amplitude for one or all of the gaussians

    mean_low: scalar or array
        Lower bound on the mean for one or all of the gaussians
    mean_high: scalar or array
        Upper bound on the mean for one or all of the gaussians

    sigma_low: scalar or array
        Lower bound on sigma for one or all of the gaussians
    sigma_high: scalar or array
        Upper bound on sigma for one or all of the gaussians

    Returns
    -------
    gmix, info

    gmix: gaussian mixture
        Best fit mixture
    info: dict
        Information on the processing

        converged: True if converged
        numiter: number of iterations performed
        loglike: log likelihood
        absdiff: abs(loglike - loglike_old) for last iteration
    """

    converged = False

    assert guess.size > 0, "guess must have size > 0"
    gmix = guess.copy()
    npoints = data.size
    ngauss = gmix.size

    constraints = extract_constraints(ngauss, constraint_kw)

    T = np.zeros((npoints, ngauss))

    loglike_old = -np.inf

    for i in range(maxiter):
        do_e_step(gmix, data, T)
        do_m_step(gmix, data, T)

        _apply_constraints(gmix=gmix, **constraints)

        loglike = get_loglike(gmix, data)

        absdiff = np.abs(loglike/loglike_old - 1)
        if absdiff < tol:
            converged = True
            break

        loglike_old = loglike

    info = {
        'converged': converged,
        'numiter': i+1,
        'loglike': loglike,
        'absdiff': absdiff,
    }
    return gmix, info


def plot_gmix(
    *,
    gmix,
    min=None,
    max=None,
    npts=None,
    data=None,
    nbin=None,
    binsize=None,
    file=None,
    dpi=100,
    show=False,
    plt=None,
    **plot_kws,
):
    """
    plot the model and each component.  Optionally plot a set of
    data as well.  Currently only works for 1d

    Parameters
    ----------
    min: float
        Min value to plot, if data is sent then this can be left
        out and the min value will be gotten from that data.
    max: float
        Max value to plot, if data is sent then this can be left
        out and the max value will be gotten from that data.
    npts: int, optional
        Number of points to use for the plot.  If data are sent you
        can leave this off and a suitable value will be chosen based
        on the data binsize
    data: array, optional
        Optional data to plot as a histogram
    nbin: int, optional
        Optional number of bins for histogramming data
    binsize: float, optional
        Optional binsize for histogramming data
    file: str, optional
        Optionally write out a plot file
    dpi: int, optional
        Optional dpi for graphics like png, default 100
    show: bool, optional
        If True, show the plot on the screen

    Returns
    -------
    plot object
    """
    import esutil as eu
    import hickory

    if plt is None:
        plt = hickory.Plot(**plot_kws)

    if data is not None:

        if min is None:
            min = data.min()
        if max is None:
            max = data.max()

        if binsize is not None:
            dx_data = binsize
        elif nbin is not None:
            dx_data = (min - max)/nbin
        else:
            raise ValueError('send either binsize or nbin when sending data')

        if npts is None:
            dx_model = dx_data/10
            npts = int((max - min)/dx_model)

        xvals = np.linspace(
            min,
            max,
            npts,
        )
        dx_model = xvals[1] - xvals[0]

        wdata, = np.where((data > min) & (data < max))

        if wdata.size == 0:
            dsum = 0.0
        else:
            hd = eu.stat.histogram(
                data, min=min, max=max, nbin=nbin, binsize=binsize, more=True,
            )
            dsum = hd['hist'].sum()

            plt.bar(hd['center'], hd['hist'], label='data', width=dx_data,
                    alpha=0.5, color='#a6a6a6')

    else:
        if npts is None:
            raise ValueError('send npts if not sending data')
        if min is None:
            raise ValueError('send min if not sending data')
        if max is None:
            raise ValueError('send max if not sending data')

        xvals = np.linspace(min, max, npts)

    predicted = gmix_eval(gmix, xvals)

    if data is not None:
        psum = predicted.sum()
        if dsum > 0:
            fac = dsum/psum * dx_data/dx_model
        else:
            fac = dx_data/dx_model
    else:
        fac = 1.0

    plt.curve(xvals, fac*predicted, label='model')
    for i in range(gmix.size):
        predicted = gauss_eval(gmix[i], xvals)

        label = 'component %d' % i
        plt.curve(xvals, fac*predicted, label=label)

    if show:
        plt.show()

    if file is not None:
        print('writing:', file)
        plt.savefig(file, dpi=dpi)

    return plt


def gmix_sample(gmix, rng, size=None, components=None):
    """
    sample from a gaussian mixture

    Parameters
    ----------
    gmix: gaussian mixture
        the mixture to sample
    rng: np.RandomState
        Random number generator
    size: int, optional
        If sent, an array of this many samples are generated and
        an array is returned.  If not sent a scalar is returned

    Returns
    -------
    samples, scalar or array
    """

    if size is None:
        scalar = True
        size = 1
    else:
        scalar = False

    if components is None:
        components = np.arange(gmix.size)

    weights = gmix['weight'][components]/gmix['weight'][components].sum()
    means = gmix['mean'][components]
    sigmas = gmix['sigma'][components]

    component_sizes = rng.multinomial(size, weights)

    samples = np.zeros(size)

    start = 0
    for mcs in zip(means, sigmas, component_sizes):
        mean, sigma, component_size = mcs
        component_samples = rng.normal(
            loc=mean,
            scale=sigma,
            size=component_size,
        )

        samples[start:start+component_size] = component_samples
        start += component_size

    if scalar:
        samples = samples[0]

    return samples


def gauss_print(gauss):
    """
    print a gaussian

    Parameters
    ----------
    gauss: gaussian
        The gaussian to print
    """
    s = '%g %g %g'
    print(s % (gauss['weight'], gauss['mean'], gauss['sigma']))


def gmix_print(gmix):
    """
    print the gaussian mixture

    Parameters
    ----------
    gmix: gaussian mixture
        The gaussian mixture to print
    """
    for i in range(gmix.size):

        gauss = gmix[i]
        s = '%d  %g %g %g' % (
            i, gauss['weight'], gauss['mean'], gauss['sigma']
        )
        print(s)


def make_gmix(n):
    """
    make a gaussian mixture with n elements

    Parameters
    ----------
    n: int
        Number of gaussians in the mixture
    """
    return np.zeros(n, dtype=GAUSS_DTYPE)


@njit
def gauss_set(gauss, weight, mean, sigma):
    """
    Set the parameters of a gaussian

    Parameters
    ----------
    gauss: gaussian
        The gaussian to set
    weight: float
        The amplitude of the gaussian
        the cluster
    mean: float
        The mean of the gaussian
    sigma: float
        The sigma of the gaussian
    """
    if sigma < MIN_SIGMA_DEFAULT:
        sigma = MIN_SIGMA_DEFAULT

    gauss['weight'] = weight
    gauss['mean'] = mean
    gauss['sigma'] = sigma
    gauss['icovar'] = 1/sigma**2
    gauss['norm'] = weight/np.sqrt(2 * np.pi)/sigma


def gauss_eval(gauss, x):
    """
    Evaluate a gaussian at a point or points

    Parameters
    ----------
    gauss: the gaussian
        The gaussian to evaluate
    x: float or array
        The position(s) at which to evaluate the gaussian

    Returns
    -------
    val(s): float
    """

    if isinstance(x, np.ndarray):
        return gauss_eval_array(gauss, x)
    else:
        return gauss_eval_scalar(gauss, x)


@njit
def gauss_eval_scalar(gauss, x):
    """
    Evaluate a gaussian at a single point

    Parameters
    ----------
    gauss: the gaussian
        The gaussian to evaluate
    x: float
        The position at which to evaluate the gaussian

    Returns
    -------
    val: float
    """
    arg = -0.5 * (gauss['mean'] - x)**2 * gauss['icovar']
    return gauss['norm'] * np.exp(arg)


@njit
def gauss_eval_array(gauss, x):
    """
    Evaluate a gaussian at a an array of points

    Parameters
    ----------
    gauss: the gaussian
        The gaussian to evaluate
    x: array
        The positions at which to evaluate the gaussian

    Returns
    -------
    arr: floats
    """

    output = np.zeros(x.size)

    for ix in range(x.size):
        output[ix] = gauss_eval_scalar(gauss, x[ix])

    return output


def gmix_eval(gmix, x):
    """
    Evaluate a gaussian mixture at a point or set of points

    Parameters
    ----------
    gmix: the gaussian mixture
        The mixture to evaluate
    x: float or array
        The position(s) at which to evaluate the mixture

    Returns
    -------
    float or arrary
    """

    if isinstance(x, np.ndarray):
        return gmix_eval_array(gmix, x)
    else:
        return gmix_eval_scalar(gmix, x)


@njit
def gmix_eval_scalar(gmix, x):
    """
    Evaluate a gaussian mixture at a point

    Parameters
    ----------
    gmix: the gaussian mixture
        The mixture to evaluate
    x: float
        The position at which to evaluate the mixture

    Returns
    -------
    val: float
    """

    val = 0.0
    for igauss in range(gmix.size):
        gauss = gmix[igauss]
        val += gauss_eval_scalar(gauss, x)

    return val


@njit
def gmix_eval_array(gmix, x):
    """
    Evaluate a gaussian mixture at a set of points

    Parameters
    ----------
    gmix: the gaussian mixture
        The mixture to evaluate
    x: array
        The positions at which to evaluate the mixture

    Returns
    -------
    arr: floats
    """

    output = np.zeros(x.size)

    for ix in range(x.size):
        output[ix] = gmix_eval_scalar(gmix, x[ix])

    return output


@njit
def gmix_get_mean(gmix):
    wsum = gmix['weight'].sum()
    return (gmix['weight']*gmix['mean']).sum()/wsum

@njit
def gmix_get_sigma(gmix):

    mean = gmix_get_mean(gmix)

    diff = gmix['mean'] - mean

    wsum = gmix['weight'].sum()
    var = ((gmix['sigma']**2 + diff ** 2) * gmix['weight']).sum() / wsum
    return np.sqrt(var)


@njit
def get_loglike(gmix, x):
    """
    get the log likelihood of the mixture over the set of points

    Parameters
    ----------
    gmix: the gaussian mixture
        The mixture to evaluate
    x: array
        The positions at which to evaluate the mixture

    Returns
    -------
    loglike: float
    """
    npoints = x.size

    loglike = 0.0
    for ix in range(npoints):
        s = gmix_eval_scalar(gmix, x[ix])
        loglike += np.log(s)

    # because we don't normalize the amplitudes
    # return loglike - np.log(x.size)
    return loglike


@njit
def do_e_step(gmix, x, T):
    """
    perform the expectation step, filling the T array

    Parameters
    ----------
    gmix: the gaussian mixture
        The mixture to evaluate
    x: array
        The positions at which to evaluate the mixture
    T: array[npoints, ngauss]
        The T array to fill

    Returns
    -------
    None
    """
    for ix in range(x.size):
        xval = x[ix]

        vsum = 0.0
        for igauss in range(gmix.size):
            gauss = gmix[igauss]

            val = gauss_eval_scalar(gauss, xval)
            # print(igauss, xval, val)
            T[ix, igauss] = val
            vsum += val

        ivsum = 1/vsum
        for igauss in range(gmix.size):
            T[ix, igauss] *= ivsum


@njit
def do_m_step(gmix, x, T):
    """
    perform the maximization step, filling the mixture with
    the new parameters

    Parameters
    ----------
    gmix: the gaussian mixture
        The mixture to evaluate
    x: array
        The positions at which to evaluate the mixture
    T: array[npoints, ngauss]
        The T array

    Returns
    -------
    None
    """

    ngauss = gmix.size
    npoints = x.size

    for igauss in range(ngauss):
        gauss = gmix[igauss]

        Tsum = 0.0
        mean_sum = 0.0
        covar_sum = 0.0

        mean = gauss['mean']

        for ix in range(npoints):
            xval = x[ix]
            Tval = T[ix, igauss]

            Tsum += Tval
            mean_sum += Tval * xval
            covar_sum += Tval*(xval - mean)**2

        weight_new = Tsum/npoints
        mean_new = mean_sum/Tsum
        covar_new = covar_sum/Tsum

        sigma_new = np.sqrt(covar_new)

        gauss_set(
            gauss,
            weight_new,
            mean_new,
            sigma_new,
        )


def _apply_constraints(
    *,
    gmix,
    weight_low,
    weight_high,
    mean_low,
    mean_high,
    sigma_low,
    sigma_high,
):
    """
    clip the mixture values according to the input constraints
    The constraints can be None, a scalar, or an array of length
    ngauss

    Parameters
    ----------
    gmix: the gaussian mixture
        The mixture to evaluate

    weight_low: scalar or array
        Lower bound on the amplitude for one or all of the gaussians
    weight_high: scalar or array
        Upper bound on the amplitude for one or all of the gaussians

    mean_low: scalar or array
        Lower bound on the mean for one or all of the gaussians
    mean_high: scalar or array
        Upper bound on the mean for one or all of the gaussians

    sigma_low: scalar or array
        Lower bound on sigma for one or all of the gaussians
    sigma_high: scalar or array
        Upper bound on sigma for one or all of the gaussians

    Returns
    -------
    None

    """

    reset = False

    if weight_low is not None or weight_high is not None:
        reset = True
        gmix['weight'] = gmix['weight'].clip(min=weight_low, max=weight_high)

    if mean_low is not None or mean_high is not None:
        reset = True
        gmix['mean'] = gmix['mean'].clip(min=mean_low, max=mean_high)

    if sigma_low is not None or sigma_high is not None:
        reset = True
        gmix['sigma'] = gmix['sigma'].clip(min=sigma_low, max=sigma_high)

    # make consistent
    if reset:
        for i in range(gmix.size):
            gauss = gmix[i]
            gauss_set(
                gauss,
                gauss['weight'],
                gauss['mean'],
                gauss['sigma'],
            )


def extract_constraint(*, ngauss, constraint, name):
    """
    extract a constraint

    Parameters
    ----------
    ngauss: int
        Number of gaussians in the mixture
    constraint: float or array
        Either None, a float or an array with length ngauss.
    name: str
        Name of the constraint for information messages

    Returns
    -------
    None or array of length ngauss
    """
    if constraint is not None:
        constraint = np.array(constraint, ndmin=1, dtype='f8')
        csize = constraint.size
        if csize < ngauss:
            if csize == 1:
                constraint = np.zeros(ngauss) + constraint
            else:
                raise ValueError(
                    'constraint %s must be size 1 or ngauss=%d'
                    'got %d' % (name, ngauss, csize)
                )

    return constraint


def extract_constraints(ngauss, kw):
    """
    extract all constraints.  The keys of of the dict should,
    if present, match those shown in

    Parameters
    -----------
    ngauss: int
        Number of gaussians in the mixture
    kw: dict
        Keys should be in this set

        weight_low: scalar or array
            Lower bound on the amplitude for one or all of the gaussians
        weight_high: scalar or array
            Upper bound on the amplitude for one or all of the gaussians

        mean_low: scalar or array
            Lower bound on the mean for one or all of the gaussians
        mean_high: scalar or array
            Upper bound on the mean for one or all of the gaussians

        sigma_low: scalar or array
            Lower bound on sigma for one or all of the gaussians
        sigma_high: scalar or array
            Upper bound on sigma for one or all of the gaussians

    Returns
    -------
    dict with all keys present
    """
    constraints = {}
    ctypes = ['weight', 'mean', 'sigma']
    for ctype in ctypes:
        for side in ['low', 'high']:
            cname = '%s_%s' % (ctype, side)

            if cname == 'sigma_low':
                default = MIN_SIGMA_DEFAULT
            else:
                default = None

            constraints[cname] = extract_constraint(
                ngauss=ngauss,
                constraint=kw.get(cname, default),
                name=cname,
            )

    return constraints


def test_1gauss():
    seed = None
    maxiter = 100

    rng = np.random.RandomState(seed)
    mean = 1.33
    sigma = 2.0

    sigma_low = 0.1*sigma

    npoints = 300
    data = rng.normal(loc=mean, scale=sigma, size=npoints)

    gmix_guess = make_gmix(1)
    gauss_set(
        gmix_guess[0],
        rng.uniform(low=0.9, high=1.1),
        rng.uniform(low=0.8*mean, high=1.2*mean),
        rng.uniform(low=0.8*sigma, high=1.2*sigma),
    )
    print(gmix_guess)

    gmix, info = run_em(data, gmix_guess, maxiter, sigma_low=sigma_low)
    print(gmix)
    print(info)

    binsize = sigma/7
    plot_gmix(
        gmix=gmix,
        data=data,
        binsize=binsize,
        show=True,
    )


def test_2gauss():
    seed = None
    maxiter = 1000

    rng = np.random.RandomState(seed)
    mean1 = 0.00
    mean2 = 1.33
    sigma1 = 0.1
    sigma2 = 1.0

    sigma_low = 0.1*min(sigma1, sigma2)

    npoints = 1000
    npoints1 = int(0.5*npoints)
    npoints2 = int(0.5*npoints)

    data = np.hstack([
        rng.normal(loc=mean1, scale=sigma1, size=npoints1),
        rng.normal(loc=mean2, scale=sigma2, size=npoints2),
    ])

    gmix_guess = make_gmix(2)
    gauss_set(
        gmix_guess[0],
        rng.uniform(low=0.4*npoints, high=0.6*npoints),
        rng.uniform(low=2, high=3),  # guess way high
        rng.uniform(low=0.8*sigma1, high=1.2*sigma1),
    )
    gauss_set(
        gmix_guess[1],
        rng.uniform(low=0.4*npoints, high=0.6*npoints),
        rng.uniform(low=0, high=0.4),
        rng.uniform(low=0.8*sigma2, high=1.2*sigma2),
    )

    mean_low = [-1, 0.5]
    mean_high = [0.4, 2.5]

    print('guess')
    gmix_print(gmix_guess)

    gmix, info = run_em(
        data, gmix_guess, maxiter,
        mean_low=mean_low, mean_high=mean_high,
        sigma_low=sigma_low,
    )
    print('best fit')
    gmix_print(gmix)
    print(info)

    binsize = min(sigma1, sigma2)/4
    _ = plot_gmix(
        gmix=gmix,
        data=data,
        binsize=binsize,
        legend=True,
        show=True,
    )

    # samples = gmix_sample(gmix, rng, size=npoints)
    # plt.hist(samples, binsize=binsize, alpha=0.5)
    # plt.show()


def test_4gauss():
    seed = 55
    maxiter = 2000

    rng = np.random.RandomState(seed)
    mean1 = 0.00
    mean2 = 0.01
    mean3 = 0.08
    mean4 = 0.15

    sigma1 = 0.005
    sigma2 = 0.01
    sigma3 = 0.04
    sigma4 = 0.08

    npoints = 3000
    npoints1 = int(0.3*npoints)
    npoints2 = int(0.1*npoints)
    npoints3 = int(0.3*npoints)
    npoints4 = int(0.3*npoints)

    data = np.hstack([
        rng.normal(loc=mean1, scale=sigma1, size=npoints1),
        rng.normal(loc=mean2, scale=sigma2, size=npoints2),
        rng.normal(loc=mean3, scale=sigma3, size=npoints3),
        rng.normal(loc=mean4, scale=sigma4, size=npoints4),
    ])

    gmix_guess = make_gmix(4)

    mean_low = [-0.001, -0.001, 0.05, 0.05]
    mean_high = [0.001, 0.02, 0.2, 0.2]
    sigma_low = 0.003

    gauss_set(
        gmix_guess[0],
        rng.uniform(low=0.1*npoints, high=0.9*npoints),
        rng.uniform(low=mean_low[0], high=mean_high[0]),
        rng.uniform(low=0.5*sigma1, high=1.5*sigma1),
    )
    gauss_set(
        gmix_guess[1],
        rng.uniform(low=0.1*npoints, high=0.9*npoints),
        rng.uniform(low=mean_low[1], high=mean_high[1]),
        rng.uniform(low=0.5*sigma2, high=1.5*sigma2),
    )
    gauss_set(
        gmix_guess[2],
        rng.uniform(low=0.1*npoints, high=0.9*npoints),
        rng.uniform(low=mean_low[2], high=mean_high[2]),
        rng.uniform(low=0.5*sigma3, high=1.5*sigma3),
    )
    gauss_set(
        gmix_guess[3],
        rng.uniform(low=0.1*npoints, high=0.9*npoints),
        rng.uniform(low=mean_low[3], high=mean_high[3]),
        rng.uniform(low=0.5*sigma4, high=1.5*sigma4),
    )

    print('guess')
    gmix_print(gmix_guess)

    gmix, info = run_em(
        data=data,
        guess=gmix_guess,
        maxiter=maxiter,
        mean_low=mean_low, mean_high=mean_high,
        sigma_low=sigma_low,
    )
    print('best fit')
    gmix_print(gmix)
    print(info)

    binsize = min(sigma1, sigma2)/4
    _ = plot_gmix(
        gmix=gmix,
        data=data,
        binsize=binsize,
        legend=True,
        show=True,
    )

    # samples = gmix_sample(gmix, rng, size=npoints)
    # plt.hist(samples, binsize=binsize, alpha=0.5)
    # plt.show()


if __name__ == '__main__':
    # test_1gauss()
    # test_2gauss()
    test_4gauss()
