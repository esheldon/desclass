import numpy as np
from numba import njit
from .star_em import make_star_gmix, star_gmix_set
from . import cem
from .cem import gauss_set, gauss_eval_scalar
from .interp import smooth_data_hann3, interp_gp
from desclass.cem import (
    gmix_get_weight,
    gmix_get_mean,
    gmix_get_sigma,
    gmix_eval_scalar,
)


def calculate_prob(pdf_data, rmag, conc, rng):
    """
    calculate the probability that each object is a star or galaxy

    Parameters
    ----------
    pdf_data: array with fields
        PDF data for bins in r magnitude.  Should have fields 'gmix' and
        'rmag', the bin centers
    rmag: array
        We will evaluate the pdf data at these rmag points
    conc: array
        We will evaluate the pdf data at these concentration points
    rng: np.RandomState
        Random state used by GaussianProcessRegressor

    Returns
    --------
    prob_gal, prob_star: arrays
        Arrays withe same size as rmag/conc
    """
    rmag = np.array(rmag, dtype='f8', copy=False)
    conc = np.array(conc, dtype='f8', copy=False)

    gmixes = pdf_data['gmix']
    ngauss = gmixes['mean'].shape[1]

    centers = pdf_data['rmag']

    # interpolate star overall weight/mean/sigma etc
    star_gmixes = gmixes[:, :3]

    weight_interp, mean_interp, sigma_interp = interpolate_star_gmix(
        rmag_centers=centers, star_gmixes=star_gmixes, rmag=rmag, rng=rng,
    )
    star_gmix = make_star_gmix(1, 0.5, 1.0)
    star_sums = np.zeros(rmag.size)
    sum_interpolated_star_gmix(
        weight_interp, mean_interp, sigma_interp,
        conc, star_sums, star_gmix,
    )

    # interpolate each gal gaussian separately
    gauss = np.zeros(1, dtype=cem.GAUSS_DTYPE)[0]
    gal_sums = np.zeros(rmag.size)

    for igauss in range(3, ngauss):

        weight_interp, mean_interp, sigma_interp = interpolate_gauss(
            rmag_centers=centers,
            gmixes=gmixes,
            igauss=igauss,
            rmag=rmag,
            rng=rng,
        )

        sum_interpolated_gauss(
            weight_interp, mean_interp, sigma_interp, conc, gal_sums,
            gauss,
        )

    tot_sums = gal_sums + star_sums

    w, = np.where(tot_sums > 0)
    prob_gal = np.zeros(rmag.size)
    prob_star = np.zeros(rmag.size)

    prob_gal[w] = gal_sums[w]/tot_sums[w]
    prob_star[w] = star_sums[w]/tot_sums[w]

    return prob_gal, prob_star


@njit
def sum_interpolated_star_gmix(weights, means, sigmas, conc, vals, star_gmix):
    """
    internal function to sun the value from the nterpolated star mixture

    Parameters
    ----------
    weights: array
        Interpolated overall star weight
    means: array
        Interpolated overall star means
    sigmas: array
        Interpolated overall star sigmas
    conc: array
        The concentrations values at which to evaluate the mixtures, same size
        as conc
    vals: array
        Array which will hold the values
    star_gmix: gmix
        Scratch gmix for calculations

    Returns
    -------
    None
    """

    for i in range(weights.size):
        star_gmix_set(
            star_gmix,
            weights[i],
            means[i],
            sigmas[i],
        )
        vals[i] += gmix_eval_scalar(star_gmix, conc[i])


@njit
def sum_interpolated_gauss(weights, means, sigmas, conc, vals, gauss):
    """
    internal function to sun the value from interpolated gaussian

    Parameters
    ----------
    weights: array
        Interpolated weight
    means: array
        Interpolated means
    sigmas: array
        Interpolated sigmas
    conc: array
        The concentrations values at which to evaluate the gaussians
    vals: array
        Array which will hold the values, same size as conc
    star_gmix: gmix
        Scratch gmix for calculations

    Returns
    -------
    None
    """

    for i in range(weights.size):
        gauss_set(
            gauss,
            weights[i],
            means[i],
            sigmas[i],
        )
        vals[i] += gauss_eval_scalar(gauss, conc[i])


def plothist_prob(prob, type, label, show=False):
    """
    plot histogram of the probabilities, with a log and a linear
    plot

    Parameters
    ----------
    prob: array
        The prob values
    type: string
        The type for labeling the x axis, e.g. for 'star'.  The x axis
        will be labelled 'star probability'
    show: bool
        If True, show on the screen

    Returns
    -------
    tab: the figure
        A matplotlib figure with the table of plots
    """
    import hickory
    tab = hickory.Table(
        nrows=2,
    )

    bins = 100
    tab[0].hist(prob, bins=bins)
    tab[0].set_title(label)

    tab[1].hist(prob, bins=bins)
    tab[1].set_yscale('log')
    tab[1].set_ylim(bottom=0.5)
    tab[1].set_xlabel(type+' probability')

    if show:
        tab.show()

    return tab


def interpolate_star_gmix(*, rmag_centers, star_gmixes, rmag, rng):
    """
    interpolate the star gmixes fit in bins of rmag

    Parameters
    ----------
    rmag_centers: array
        Centers of the rmag bins
    star_gmixes: gmix array
        e.g. pdf_data['gmix'][:, :3]
    rmag: array
        rmag positions at which to interpolate
    rng: np.RandomState
        Random state used by GaussianProcessRegressor

    Returns
    -------
    weight_interp, mean_interp, sigma_interp
        arrays with same size as rmag
    """
    # centers = data['rmag']
    # star_gmixes = data['gmix'][:, :3]

    weights = np.array([gmix_get_weight(gm) for gm in star_gmixes])
    means = np.array([gmix_get_mean(gm) for gm in star_gmixes])
    sigmas = np.array([gmix_get_sigma(gm) for gm in star_gmixes])

    ystd = (smooth_data_hann3(weights) - weights).std()
    weight_interp, ysigma, weight_gp = interp_gp(
        rmag_centers, weights, ystd, rmag, rng=rng,
    )

    ystd = (smooth_data_hann3(means) - means).std()
    mean_interp, ysigma, means_gp = interp_gp(
        rmag_centers, means, ystd, rmag, rng=rng,
    )

    ystd = (smooth_data_hann3(sigmas) - sigmas).std()
    ysend = smooth_data_hann3(sigmas)

    sigma_interp, ysigma, sigma_gp = interp_gp(
        rmag_centers, ysend, ystd, rmag, rng=rng,
    )

    return weight_interp, mean_interp, sigma_interp


def interpolate_gauss(*, rmag_centers, gmixes, igauss, rmag, rng):
    """
    interpolate a single gaussian from pdf fits in bins of rmag

    Parameters
    ----------
    rmag_centers: array
        Centers of the rmag bins
    star_gmixes: gmix array
        e.g. pdf_data['gmix']
    igauss: int
        the gaussian in the mixture to interpolate
    rmag: array
        rmag positions at which to interpolate
    rng: np.RandomState
        Random state used by GaussianProcessRegressor

    Returns
    -------
    weight_interp, mean_interp, sigma_interp
        arrays with same size as rmag
    """

    # centers = data['rmag']
    # star_gmixes = data['gmix'][:, :3]

    weights = gmixes['weight'][:, igauss]
    means = gmixes['mean'][:, igauss]
    sigmas = gmixes['sigma'][:, igauss]

    ystd = (smooth_data_hann3(weights) - weights).std()
    weight_interp, ysigma, weight_gp = interp_gp(
        rmag_centers, weights, ystd, rmag, rng=rng,
    )

    ystd = (smooth_data_hann3(means) - means).std()
    mean_interp, ysigma, means_gp = interp_gp(
        rmag_centers, means, ystd, rmag, rng=rng,
    )

    ystd = (smooth_data_hann3(sigmas) - sigmas).std()
    ysend = smooth_data_hann3(sigmas)

    sigma_interp, ysigma, sigma_gp = interp_gp(
        rmag_centers, ysend, ystd, rmag, rng=rng,
    )

    return weight_interp, mean_interp, sigma_interp
