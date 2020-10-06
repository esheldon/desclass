import numpy as np
from esutil.numpy_util import between
from .interp import interpolate_gauss


def calculate_purity(*, pdf_data, rmag, conc, rng):
    """
    Calculate the star and galaxy purity at the input mag and concentration
    values

    star_purity = intergral(nstar)/(integral(nstar) + integral(ngal)) from left
    gal_purity intergral(ngal)/(integral(nstar) + integral(ngal)) from righ

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
    purity_gal, purity_star: arrays
        Arrays with same size as input rmag/conc
    """
    import scipy.stats

    rmagmin = pdf_data['rmagmin'][0]
    rmagmax = pdf_data['rmagmax'][-1]
    print('limiting to:', rmagmin, rmagmax)

    centers = pdf_data['rmag']
    gmixes = pdf_data['gmix']
    ngauss = gmixes['mean'].shape[1]

    all_cdf = np.zeros(rmag.size)
    star_cdf = np.zeros(rmag.size)

    all_sf = np.zeros(rmag.size)
    gal_sf = np.zeros(rmag.size)

    for igauss in range(ngauss):
        weight, mean, sigma = interpolate_gauss(
            rmag_centers=centers,
            gmixes=gmixes,
            igauss=igauss,
            rmag=rmag,
            rng=rng,
        )

        # cdf, cumulative integral from left for star purities
        cdf_vals = weight*scipy.stats.norm.cdf(conc, loc=mean, scale=sigma)
        all_cdf += cdf_vals

        # sf, cumulative integral from right for gal purities
        sf_vals = weight*scipy.stats.norm.sf(conc, loc=mean, scale=sigma)
        all_sf += sf_vals

        if igauss < 3:
            star_cdf += cdf_vals
        else:
            gal_sf += sf_vals

    star_purity = np.repeat(-9.999e9, rmag.size)
    gal_purity = np.repeat(-9.999e9, rmag.size)

    w, = np.where(
        between(rmag, rmagmin, rmagmax) &
        (all_cdf > 0)
    )
    star_purity[w] = star_cdf[w]/all_cdf[w]

    w, = np.where(
        between(rmag, rmagmin, rmagmax) &
        (all_sf > 0)
    )
    gal_purity[w] = gal_sf[w]/all_sf[w]

    return gal_purity, star_purity


def plot_purity(*, pdf_data, type, show=False):
    """
    plot the cumulative contamination, e.g.

    intergral(nstar)/(integral(nstar) + integral(ngal))
    intergral(ngal)/(integral(nstar) + integral(ngal))

    for the binned pdf data
    """
    import scipy.stats
    import hickory

    tab = hickory.Table(
        figsize=(10, 7.5),
        nrows=4, ncols=4,
    )

    gmixes = pdf_data['gmix']
    ngauss = gmixes['mean'].shape[1]
    npts = 1000

    if type == 'star':
        minconc, maxconc = -0.01, 0.005
    else:
        minconc, maxconc = -0.001, 0.005

    num = 1000
    conc = np.linspace(minconc, maxconc, num)

    for rmagbin in range(pdf_data.size):

        label = r'$%.2f < r < %.2f$' % (
            pdf_data['rmagmin'][rmagbin],
            pdf_data['rmagmax'][rmagbin],
        )

        all_cdf = np.zeros(npts)
        this_cdf = np.zeros(npts)
        purity = np.zeros(npts)

        for igauss in range(ngauss):
            mean = gmixes['mean'][rmagbin, igauss]
            sigma = gmixes['sigma'][rmagbin, igauss]
            norm = gmixes['num'][rmagbin, igauss]

            n = scipy.stats.norm(loc=mean, scale=sigma)

            if type == 'star':
                vals = norm * n.cdf(conc)
            else:
                vals = norm * n.sf(conc)

            all_cdf += vals

            if type == 'star' and igauss < 3:
                this_cdf += vals
            elif type == 'gal' and igauss >= 3:
                this_cdf += vals

        w, = np.where(all_cdf > 0)
        purity[w] = this_cdf[w]/all_cdf[w]

        ylim = (
            0.5*purity[w].max(),
            1.1,
        )

        if type == 'star':
            xlim = [
                -0.001,
                maxconc,
            ]
        else:
            xlim = [minconc, maxconc]

        ax = tab.axes[rmagbin]
        ax.set(
            title=label,
            xlabel='concentration',
            ylabel='%s purity' % type,
            xlim=xlim,
            ylim=ylim,
        )
        ax.axhline(1, color='black')
        ax.curve(conc[w], purity[w])

    if show:
        tab.show()

    return tab
