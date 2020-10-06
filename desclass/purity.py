import numpy as np


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
