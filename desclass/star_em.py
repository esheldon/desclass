import numpy as np
from numba import njit
from desclass import cem
from desclass.cem import (
    gauss_set,
    make_gmix,
    make_gauss,
    gmix_get_mean,
    gmix_get_sigma,
    get_loglike,
    do_e_step,
)
import esutil as eu


class StarEMFitter(object):
    def __init__(
        self,
        *,
        data,
        star_constraints,
        gal_constraints,
        rng,
    ):

        self.rng = rng
        # numba can't deal with '>f8'
        self.data = np.array(data, copy=False, dtype='f8')
        self.maxiter = 1000
        self.tol = 1.0e-6

        self.star_constraints = star_constraints
        self.gal_constraints = gal_constraints
        if self.star_constraints is None:
            raise ValueError('send star constraints')

        self.gal_ngauss = len(gal_constraints)

    def go(self, ntry=100):

        for i in range(ntry):
            self.guess = self.make_guess()

            print('guess')
            cem.gmix_print(self.guess)

            gmix, self.result = run_sg_em(
                data=self.data,
                guess=self.guess,
                maxiter=self.maxiter,
                tol=self.tol,
                star_constraints=self.star_constraints,
                gal_constraints=self.gal_constraints,
            )

            if self.result['converged']:
                break

        self.gmix = self._add_num(gmix)
        if self.result['converged']:
            self.result['flags'] = 0
        else:
            self.result['flags'] = 1

    def _add_num(self, gmix):
        new_gmix = eu.numpy_util.add_fields(gmix, [('num', 'f8')])
        new_gmix['num'] = new_gmix['weight'] * self.data.size
        return new_gmix

    def make_guess(self):
        sc = self.star_constraints

        glist = []

        weight = guess_from_constraint(
            rng=self.rng,
            low=sc['weight_low'],
            high=sc['weight_high'],
        )
        mean = guess_from_constraint(
            rng=self.rng,
            low=sc['mean_low'],
            high=sc['mean_high'],
        )
        sigma = guess_from_constraint(
            rng=self.rng,
            low=sc['sigma_low'],
            high=sc['sigma_high'],
        )
        tguess = make_star_gmix(weight, mean, sigma)
        glist.append(tguess)

        if self.gal_constraints is not None:
            for gc in self.gal_constraints:

                weight = guess_from_constraint(
                    rng=self.rng,
                    low=gc['weight_low'],
                    high=gc['weight_high'],
                )
                mean = guess_from_constraint(
                    rng=self.rng,
                    low=gc['mean_low'],
                    high=gc['mean_high'],
                )
                sigma = guess_from_constraint(
                    rng=self.rng,
                    low=gc['sigma_low'],
                    high=gc['sigma_high'],
                )
                tguess = make_gauss(weight, mean, sigma)
                glist.append(tguess)

        return np.hstack(glist)

    def plot(
        self, *,
        min=None,
        max=None,
        nbin=None,
        binsize=0.0004,
        figsize=(10, 7.5),
        file=None,
        dpi=100,
        show=False,
        plt=None,
        **plot_kws
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

        return cem.plot_gmix(
            gmix=self.gmix,
            data=self.data,
            min=min,
            max=max,
            nbin=nbin,
            binsize=binsize,
            file=file,
            dpi=dpi,
            show=show,
            plt=plt,
            **plot_kws
        )

    def plot3(
        self, *,
        label=None,
        show=False,
        file=None,
        dpi=100,
        **plot_kws
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

        import hickory
        tab = hickory.Table(
            figsize=(10, 7.5),
            nrows=2,
            ncols=2,
        )
        tab[1, 1].axis('off')
        tab[0, 0].set(xlabel='concentration')
        tab[0, 1].set(xlabel='concentration')
        tab[1, 0].set(xlabel='concentration')

        tab[1, 1].ntext(0.5, 0.5, label,
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=16)
        binsize = 0.0004
        binsize_coarse = 0.0004 * 2
        maxconc = 0.025

        self.plot(
            binsize=binsize,
            xlabel='concentration',
            legend=True,
            plt=tab[0, 0],
            show=False,
        )
        tab[0, 0].legend()

        star_samples = cem.gmix_sample(
            self.gmix,
            self.rng,
            size=1000,
            components=[0, 1],
        )

        smn, ssig = eu.stat.sigma_clip(star_samples)
        xlim = (smn - 4*ssig, smn + 4*ssig)
        star_binsize = ssig/5

        tab[0, 1].set(xlim=xlim)
        self.plot(
            binsize=star_binsize,
            legend=True,
            xlim=xlim,
            plt=tab[0, 1],
            show=False,
        )

        cmin, cmax = 0.0005, maxconc
        self.plot(
            binsize=binsize_coarse,
            min=cmin,
            max=cmax,
            xlabel='concentration',
            legend=True,
            plt=tab[1, 0],
            show=False,
            xlim=(cmin, cmax),
        )

        if show:
            tab.show()

        if file is not None:
            print('writing:', file)
            tab.savefig(file, dpi=dpi)

        return tab


#
# this code should be in some other module
#

CONSTRAINTS_DTYPE = [
    ('weight_low', 'f8'),
    ('weight_high', 'f8'),
    ('mean_low', 'f8'),
    ('mean_high', 'f8'),
    ('sigma_low', 'f8'),
    ('sigma_high', 'f8'),
]


def make_constraints(
    *,
    weight_low=0, weight_high=np.inf,
    mean_low=-np.inf, mean_high=np.inf,
    sigma_low=0, sigma_high=np.inf,
    size=None,
):

    if size is None:
        scalar = True
        size = 1
    else:
        scalar = False

    c = np.zeros(size, dtype=CONSTRAINTS_DTYPE)

    if scalar:
        c = c[0]

    c['weight_low'] = weight_low
    c['weight_high'] = weight_high
    c['mean_low'] = mean_low
    c['mean_high'] = mean_high
    c['sigma_low'] = sigma_low
    c['sigma_high'] = sigma_high

    return c


@njit
def apply_constraints(constraints, weight, mean, sigma):

    if weight < constraints['weight_low']:
        weight = constraints['weight_low']
    elif weight > constraints['weight_high']:
        weight = constraints['weight_high']

    if mean < constraints['mean_low']:
        mean = constraints['mean_low']
    elif mean > constraints['mean_high']:
        mean = constraints['mean_high']

    if sigma < constraints['sigma_low']:
        sigma = constraints['sigma_low']
    elif sigma > constraints['sigma_high']:
        sigma = constraints['sigma_high']

    return weight, mean, sigma


def guess_from_constraint(*, rng, low, high, width=0.05):

    alow = low - low
    ahigh = high - low
    mid = 0.5*(ahigh + alow)

    return low + rng.uniform(
        low=(1-width/2)*mid,
        high=(1-width/2)*mid,
    )


@njit
def star_gmix_set(gmix, weight, mean, sigma):
    """
    set a 3 gaussian mixture specialized for the skewed
    stellar locus in concentration

    Parameters
    ----------
    gmix: array
        The gaussian mixture array
    mean: float
        Mean of the mixture
    sigma: float
        Sigma of the mixture

    Returns
    -------
    None
    """
    gauss_set(
        gmix[0],
        0.60*weight,
        -0.3387430272763309 * sigma + mean,
        0.5008036637302701 * sigma,
    )
    gauss_set(
        gmix[1],
        0.30*weight,
        0.08925543995352567 * sigma + mean,
        0.8860469994955699 * sigma,
    )
    gauss_set(
        gmix[2],
        0.10*weight,
        1.764691843797408 * sigma + mean,
        1.520992554551711 * sigma,
    )


def _generate_star_pdf_pars():
    """
    iterate to get exactly sigma == 1
    """
    gmix = make_gmix(3)
    sigfacs = [0.500887959032764, 0.886196138979275, 1.521248568109045]

    def set_gmix(sigma_try):
        gauss_set(
            gmix[0],
            0.60,
            -0.3387430272763309 * sigma_try,
            sigfacs[0] * sigma_try,
        )
        gauss_set(
            gmix[1],
            0.30,
            0.08925543995352567 * sigma_try,
            sigfacs[1] * sigma_try,
        )
        gauss_set(
            gmix[2],
            0.10,
            1.764691843797408 * sigma_try,
            sigfacs[2] * sigma_try,
        )

    set_gmix(1)

    # sigma_meas = np.inf
    sigma_meas = gmix_get_sigma(gmix)

    ntry = 0
    while abs(sigma_meas - 1.0) > 1.0e-17:
        set_gmix(1/sigma_meas)
        sigfacs[:] = gmix['sigma']

        set_gmix(1)
        sigma_meas = gmix_get_sigma(gmix)

        ntry += 1

    print(ntry)
    print(sigfacs)


def make_star_gmix(weight, mean, sigma):
    """
    make a 3 gaussian mixture specialized for the skewed
    stellar locus in concentration

    Parameters
    ----------
    mean: float
        Mean of the mixture
    sigma: float
        Sigma of the mixture

    Returns
    -------
    the mixture array
    """

    gmix = make_gmix(3)
    star_gmix_set(gmix, weight, mean, sigma)
    return gmix


def run_sg_em(
    *,
    data,
    guess,
    maxiter,
    tol=1.0e-6,
    star_constraints=None,
    gal_constraints=None,
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

    ngauss = guess.size

    if star_constraints is None:
        star_constraints = make_constraints()

    if gal_constraints is None:
        ngal_gauss = ngauss - 3
        if ngal_gauss > 0:
            gal_constraints = make_constraints(size=ngal_gauss)

    converged = False

    gmix = guess.copy()
    npoints = data.size

    T = np.zeros((npoints, ngauss))

    loglike_old = -np.inf

    for i in range(maxiter):
        do_e_step(gmix, data, T)
        do_sg_m_step(gmix, data, T, star_constraints, gal_constraints)

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


@njit
def do_star_m_step(gmix, x, T, star_constraints):
    """
    perform the maximization step for the special star mixture
    filling the mixture with the new parameters

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

    # should be 3
    ngauss = gmix.size
    npoints = x.size

    Tsum = 0.0
    mean_sum = 0.0
    covar_sum = 0.0

    mean = gmix_get_mean(gmix)

    for igauss in range(ngauss):
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

    weight_new, mean_new, sigma_new = apply_constraints(
        star_constraints,
        weight_new, mean_new, sigma_new,
    )

    star_gmix_set(
        gmix,
        weight_new,
        mean_new,
        sigma_new,
    )


@njit
def do_sg_m_step(gmix, x, T, star_constraints, gal_constraints):
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

    do_star_m_step(gmix[0:3], x, T, star_constraints)

    for igauss in range(3, ngauss):
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

        weight_new, mean_new, sigma_new = apply_constraints(
            gal_constraints[igauss-3],
            weight_new, mean_new, sigma_new,
        )

        gauss_set(
            gauss,
            weight_new,
            mean_new,
            sigma_new,
        )
