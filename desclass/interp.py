import numpy as np
from numba import njit
from desclass.cem import (
    gmix_get_weight,
    gmix_get_mean,
    gmix_get_sigma,
)


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


@njit
def smooth_data_hann3(y):
    """
    smooth data using 3 point hann interpolation.  The end points are not
    modified

    Parameters
    ----------
    y: array
        array of values to smooth

    Returns
    -------
    sy: array
        The smoothed array
    """
    window = np.array([0.5, 1., 0.5])
    num = window.size
    offsets = np.array([-1, 0, 1])

    sy = y.copy()

    for idata in range(1, y.size-1):
        ssum = 0.0
        ksum = 0.0

        for i in range(num):
            off = offsets[i]
            k = window[i]

            ssum += k*y[idata + off]
            ksum += k

        sy[idata] = ssum/ksum

    return sy


def fit_gp(x, y, yerr, rng=None):
    """
    fit the data with a Gaussian process

    Parameters
    ----------
    x: array
        Array of x values
    y: array
        Array of y values
    yerr: array
        Array of yerr values
    rng: np.random.RandomState, optional
        The random number generator used for the Gaussian process fitter

    Returns
    -------
    gp:  sklearn.gaussian_process.GaussianProcessRegressor
    """
    from sklearn import gaussian_process
    from sklearn.gaussian_process.kernels import (
        Matern, WhiteKernel,
        # ConstantKernel,
    )

    spacing = x[1] - x[0]
    kernel = (
        # ConstantKernel() +
        Matern(length_scale=spacing, nu=5/2) +
        WhiteKernel(noise_level=yerr**2)
    )

    gp = gaussian_process.GaussianProcessRegressor(
        kernel=kernel, normalize_y=True,
        random_state=rng,
    )

    X = x.reshape(-1, 1)
    gp.fit(X, y)

    return gp


def interp_gp(x, y, yerr, xinterp, rng=None):
    """
    interpolate data using a Gaussian process

    Parameters
    ----------
    x: array
        Array of x values
    y: array
        Array of y values
    yerr: array
        Array of yerr values
    xinterp: array
        Values at which to interpolate
    rng: np.random.RandomState, optional
        The random number generator used for the Gaussian process fitter

    Returns
    -------
    a tuple (y_pred, sigma, gp)

    y_pred: array
        Predicted values at the xinterp points
    sigma: array
        expected 1-sigma error on interpolated values
    gp: sklearn.gaussian_process.GaussianProcessRegressor
        The regressor used to predict the values
    """
    gp = fit_gp(x, y, yerr, rng=rng)

    y_pred, sigma = gp.predict(
        xinterp.reshape(-1, 1),
        return_std=True,
    )

    return y_pred, sigma, gp
