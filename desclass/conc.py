import numpy as np
from numba import njit
import ngmix
from ngmix.gmix_nb import gmix_eval_pixel
from .util import get_mag
from .constants import MAXRAD, DILATE
from .fitting import fit_gauss_am


def do_sums_oneband(
    *,
    args, rng, obslist,
    data, index, iband,
):
    """
    do the the concentration sums Observations in
    the input obslist.  also set the psf flux

    Parameters
    ----------
    args: args from argparse
        Returned from get_args()
    rng: np.RandomState
        The random number generator
    obslist: ngmix.ObsList
        Observations to measure
    """

    for obs in obslist:
        gres = get_gap_fluxes(
            args=args, rng=rng, obs=obs,
            maxrad=MAXRAD,
        )

        # flags are only set for the psf fitting
        if gres['flags'] == 0:

            data['nuse'][index] += 1
            data['flux1'][index] += gres['flux1']
            data['pflux1'][index] += gres['pflux1']

            data['flux2'][index] += gres['flux2']
            data['pflux2'][index] += gres['pflux2']


def get_gap_fluxes(*, args, rng, obs, maxrad):
    """
    get gap fluxes for a single band

    Parameters
    ----------
    args: args from argpars
        Returned from get_args()
    rng: np.RandomState
        The random number generator
    obs: ngmix.Observation
        Observation for which to get the gaussian aperture flux
    maxrad: float
        Maximum radius in arcsec

    Returns
    -------
    gres: dict
        Dict with
            flags
            flux1
            psflux1
            flux2
            psflux2
            wt
    """

    gres = {'flags': 0}

    flags = 0
    if not obs.psf.has_gmix():
        try:
            fitter = fit_gauss_am(rng=rng, obs=obs.psf)
            Tres = fitter.get_result()
            flags = Tres['flags']

            gmix = fitter.get_gmix()
            obs.psf.set_gmix(gmix)
        except ngmix.GMixRangeError:
            flags = 1
    else:
        gmix = obs.psf.gmix

    if flags != 0:
        gres['flags'] = 1
    else:

        g1, g2, T = gmix.get_g1g2T()
        Td = T*DILATE

        gres['flux1'] = get_gap_flux(obs=obs, T=T, g1=g1, g2=g2, maxrad=maxrad)
        gres['pflux1'] = get_gap_flux(obs=obs.psf, T=T, g1=g1, g2=g2,
                                      maxrad=maxrad)

        gres['flux2'] = get_gap_flux(obs=obs, T=Td, g1=g1, g2=g2,
                                     maxrad=maxrad)
        gres['pflux2'] = get_gap_flux(obs=obs.psf, T=Td, g1=g1, g2=g2,
                                      maxrad=maxrad)

    return gres


def get_gap_flux(*, obs, T, g1, g2, maxrad, get_ivar=False):
    """
    get the gaussian aperture flux

    Parameters
    ----------
    obs: ngmix.Observation
        Observation for which to get the gaussian aperture flux
    T: float
        T for weight
    g1: float
        g1 for weight
    g2: float
        g2 for weight
    maxrad: float
        Maximum radius in arcsec
    get_ivar: bool
        If True, also get the ivar_sum

    Returns
    -------
    flux_sum: float
        Sum of weight * flux.  If get_ivar is True, then (flux_sum, ivar_sum)
    """

    wt = get_weight_gmix(T=T, g1=g1, g2=g2)

    flux_sum, ivar_sum = get_weighted_sums(
        wt.get_data(),
        obs.pixels,
        maxrad,
    )
    if get_ivar:
        return flux_sum, ivar_sum
    else:
        return flux_sum


@njit
def get_weighted_sums(wt, pixels, maxrad):
    """
    get sum weight(x, y) * flux(x, y)

    weight gaussian should be normalized to 1 at the origin

    note pixels with weight <= 0 are not included in the pixel list, so
    no check for ivar > 0 is needed
    """

    flux_sum = 0.0
    # flux_var_sum = 0.0
    ivar_sum = 0.0

    maxrad2 = maxrad ** 2

    vcen = wt["row"][0]
    ucen = wt["col"][0]

    n_pixels = pixels.size
    for i_pixel in range(n_pixels):

        pixel = pixels[i_pixel]

        vmod = pixel["v"] - vcen
        umod = pixel["u"] - ucen

        rad2 = umod * umod + vmod * vmod
        if rad2 < maxrad2:

            ierr = pixel["ierr"]
            ivar = ierr * ierr
            # var = 1.0 / ivar

            weight = gmix_eval_pixel(wt, pixel)

            wdata = weight * pixel["val"]
            w2 = weight * weight

            flux_sum += wdata
            # flux_var_sum += w2 * var
            ivar_sum += w2*ivar

    return flux_sum, ivar_sum


def get_weight_gmix(*, T, g1, g2):
    """
    get the weight gaussian, normalized such that
    the value at the origin is 1

    Parameters
    ----------
    T: float
        The T value for the gaussian
    g1: float
        The g1 value for the gaussian
    g2: float
        The g2 value for the gaussian

    Returns
    -------
    ngmix.GMixModel of type gauss
    """
    pars = [0.0, 0.0, g1, g2, T, 1.0]

    wt = ngmix.GMixModel(pars, "gauss")

    wt.set_norms()
    norm = wt.get_data()['norm'][0]
    wt.set_flux(1.0/norm)
    wt.set_norms()

    return wt


def finalize_conc(*, data):
    """
    finalize the concentration calculations based on sums over all bands

    Parameters
    ----------
    data: array with fields
        The data with fields nuse, flags, flux1, flux2, pflux1, pflux2, conc

    Returns
    -------
    None
    """
    wbad, = np.where(data['nuse'] == 0)

    if wbad.size > 0:
        data['flags'][wbad] = 1
        data['flux1'][wbad] = -9999.0
        data['flux2'][wbad] = -9999.0
        data['conc'][wbad] = -9999.0

    w, = np.where(data['nuse'] > 0)

    if w.size > 0:
        nuse = data['nuse'][w]

        data['flux1'][w] *= 1.0/nuse
        data['flux2'][w] *= 1.0/nuse
        data['pflux1'][w] *= 1.0/nuse
        data['pflux2'][w] *= 1.0/nuse

        mag1 = get_mag(data['flux1'][w])
        pmag1 = get_mag(data['pflux1'][w])

        mag2 = get_mag(data['flux2'][w])
        pmag2 = get_mag(data['pflux2'][w])

        obj_conc = mag1 - mag2
        pconc = pmag1 - pmag2

        data['conc'][w] = obj_conc - pconc
