import numpy as np
import fitsio
import esutil as eu
from esutil.numpy_util import between
import hickory
import ngmix
from matplotlib.backends.backend_pdf import PdfPages
# import scipy.optimize
# from ngmix.fitting import run_leastsq
from ngmix import print_pars
# from glob import glob


def fit_gmm(*, rng, data, ngauss, min_covar=1.0e-12,
            show=False, output=None, label=None):

    gmm = ngmix.gmix_ndim.GMixND(rng=rng)
    gmm.fit(
        data,
        ngauss,
        min_covar=min_covar,
        rng=rng,
    )

    std = data.std()
    binsize = std/7

    plt = gmm.plot(
        data=data, binsize=binsize, title=label,
        legend=True,
        show=show,
        file=output,
    )

    return gmm, plt


def fit_stars_constrained(
    *, rng, data, ngauss, min_covar=1.0e-12,
    show=False, output=None, label=None
):
    from . import cem

    mn, sig = eu.stat.sigma_clip(data)

    assert ngauss == 3
    guess = cem.make_gmix(ngauss)
    cem.gauss_set(
        guess[0],
        0.6,
        mn - 0.1*abs(mn),
        sig*0.9,
    )
    cem.gauss_set(
        guess[1],
        0.3,
        mn + 0.1*abs(mn),
        sig*1.1,
    )
    cem.gauss_set(
        guess[2],
        0.1,
        mn + 2*abs(mn),
        sig*1.5,
    )

    width = 0.01
    lfac = 1 - width
    hfac = 1 + width

    weight_low = np.array([
        guess['weight'][0]*lfac,
        guess['weight'][1]*lfac,
        guess['weight'][2]*lfac,
    ])
    weight_high = np.array([
        guess['weight'][0]*hfac,
        guess['weight'][1]*hfac,
        guess['weight'][2]*hfac,
    ])

    gmm, info = cem.run_em(
        data=np.array(data, copy=False, dtype='f8'),
        guess=guess,
        maxiter=2000,
        sigma_low=1.0e-5,
        sigma_high=0.005,
        weight_low=weight_low,
        weight_high=weight_high,
    )

    std = data.std()
    binsize = std/7

    plt = cem.plot_gmix(
        gmix=gmm,
        data=data,
        binsize=binsize, title=label,
        legend=True,
        show=show,
        file=output,
    )

    return gmm, plt


def fit_gals_constrained(
    *, rng, data, ngauss, min_covar=1.0e-12,
    show=False, output=None, label=None
):
    from . import cem

    mn, sig = eu.stat.sigma_clip(data)

    assert ngauss == 2
    guess = cem.make_gmix(ngauss)
    cem.gauss_set(
        guess[0],
        0.58,
        0.8*mn,
        sig*0.4,
    )
    cem.gauss_set(
        guess[1],
        0.42,
        1.2*mn,
        sig*0.6,
    )

    width = 0.01
    lfac = 1 - width
    hfac = 1 + width
    weight_low = np.array([
        guess['weight'][0]*lfac,
        guess['weight'][1]*lfac,
    ])
    weight_high = np.array([
        guess['weight'][0]*hfac,
        guess['weight'][1]*hfac,
    ])

    gmm, info = cem.run_em(
        data=np.array(data, copy=False, dtype='f8'),
        guess=guess,
        maxiter=2000,
        sigma_low=0.001,
        sigma_high=0.02,
        weight_low=weight_low,
        weight_high=weight_high,
    )

    std = data.std()
    binsize = std/7

    plt = cem.plot_gmix(
        gmix=gmm,
        data=data,
        binsize=binsize, title=label,
        legend=True,
        show=show,
        file=output,
    )

    return gmm, plt


def test_2gauss():

    cen1 = 0
    sigma1 = 0.3
    num1 = 10000
    cen2 = 3
    sigma2 = 1
    num2 = 10000

    rng = np.random.RandomState()
    vals = [
        rng.normal(loc=cen1, scale=sigma1, size=num1),
        rng.normal(loc=cen2, scale=sigma2, size=num2),
    ]
    vals = np.hstack(vals)

    sigma = vals.std()
    binsize = sigma/10

    gmm = fit_gmm(
        data=vals,
        ngauss=2,
    )
    gmm.plot(data=vals, binsize=binsize, show=True)


def test_1gauss():

    rng = np.random.RandomState()
    num = 10000
    vals = rng.normal(size=num)

    sigma = vals.std()
    binsize = sigma/10

    gmm = fit_gmm(
        data=vals,
        ngauss=2,
    )
    gmm.plot(data=vals, binsize=binsize, show=True)


def select_stars(*, data, rmagmin, rmagmax, rmag_index):
    w, = np.where(
        (data['ext_combo'] == 0) &
        between(data['psf_mag'][:, rmag_index], rmagmin, rmagmax)
    )

    mn, std = eu.stat.sigma_clip(data['conc'][w])
    minconc, maxconc = [mn - 5*std, mn + 6*std]
    ww, = np.where(
        between(data['conc'][w], minconc, maxconc)
    )
    w = w[ww]
    return w


def select_gals(*, data, rmagmin, rmagmax, rmag_index):
    w, = np.where(
        (data['ext_combo'] == 1) &
        between(data['psf_mag'][:, rmag_index], rmagmin, rmagmax)
    )

    mn, std = eu.stat.sigma_clip(data['conc'][w])
    minconc, maxconc = [mn - 5*std, mn + 5*std]
    if rmagmin < 20:
        minconc = 0.004
    elif rmagmin < 20.5:
        minconc = 0.003
    elif rmagmin < 21:
        minconc = 0.0015
    elif rmagmin < 21.5:
        minconc = 0.001

    if minconc < -0.014:
        minconc = -0.014

    ww, = np.where(
        between(data['conc'][w], minconc, maxconc)
    )
    w = w[ww]
    return w


def get_struct(*, size, star_ngauss, gal_ngauss):
    dt = [
        ('rmagmin', 'f8'),
        ('rmagmax', 'f8'),
        ('rmag_centers', 'f8'),

        ('star_weights', 'f8', star_ngauss),
        ('star_means', 'f8', star_ngauss),
        ('star_sigmas', 'f8', star_ngauss),

        ('gal_weights', 'f8', gal_ngauss),
        ('gal_means', 'f8', gal_ngauss),
        ('gal_sigmas', 'f8', gal_ngauss),
    ]
    return np.zeros(size, dtype=dt)


def pack_struct(*, rmagmin, rmagmax, star_gmm, gal_gmm):

    if isinstance(star_gmm, np.ndarray):
        s = (-star_gmm['weight']).argsort()
        star_weights = star_gmm['weight'][s]
        star_means = star_gmm['mean'][s]
        star_sigmas = star_gmm['sigma'][s]
    else:
        s = (-star_gmm.weights.ravel()).argsort()
        star_weights = star_gmm.weights.ravel()[s]
        star_means = star_gmm.means.ravel()[s]
        star_sigmas = np.sqrt(star_gmm.covars.ravel()[s])

    if isinstance(gal_gmm, np.ndarray):
        s = (-gal_gmm['weight']).argsort()
        gal_weights = gal_gmm['weight'][s]
        gal_means = gal_gmm['mean'][s]
        gal_sigmas = gal_gmm['sigma'][s]
    else:
        s = (-gal_gmm.weights.ravel()).argsort()
        gal_weights = gal_gmm.weights.ravel()[s]
        gal_means = gal_gmm.means.ravel()[s]
        gal_sigmas = np.sqrt(gal_gmm.covars.ravel()[s])

    struct = get_struct(
        size=1,
        star_ngauss=star_means.size,
        gal_ngauss=gal_means.size,
    )

    struct['rmagmin'] = rmagmin
    struct['rmagmax'] = rmagmax
    struct['rmag_centers'] = 0.5*(rmagmax + rmagmin)

    struct['star_weights'][0] = star_weights
    struct['star_means'][0] = star_means
    struct['star_sigmas'][0] = star_sigmas

    struct['gal_weights'][0] = gal_weights
    struct['gal_means'][0] = gal_means
    struct['gal_sigmas'][0] = gal_sigmas

    return struct


def plot_all_scaled(*, struct, type, show=False, output=None):
    """
    plot scaled parameters vs rmag

    weight
    mean/sigma
    sigma/sigma_tot
    """
    if type == 'star':
        label = 'stars'
    else:
        label = 'galaxies'

    tab = hickory.Table(
        figsize=(11, 11*0.618),
        nrows=2,
        ncols=2,
    )

    tab[1, 1].axis('off')

    tab[0, 0].set(
        xlabel='r mag',
        ylabel='weight',
    )
    tab[0, 1].set(
        xlabel='r mag',
        ylabel=r'mean/$\sigma_{\mathrm{tot}}$',
    )

    tab[1, 0].set(
        xlabel='r mag',
        ylabel=r'$\sigma/\sigma_{\mathrm{tot}}$',
    )

    if type == 'star':
        label = 'stars'
    else:
        label = 'galaxies'
    tab[1, 1].ntext(
        0.5, 0.5, label,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16,
    )

    centers = struct['rmag_centers']
    ngauss = struct['%s_weights' % type].shape[1]

    # get overall sigma
    tw = struct['%s_weights' % type]
    tm = struct['%s_means' % type]
    tv = struct['%s_sigmas' % type]**2
    wsum = tw.sum(axis=1)

    mean_tot = (tw*tm).sum(axis=1)/wsum

    print('tm shape:', tm.shape)
    print('mean_tot shape:', mean_tot.shape)
    diff = tm.copy()
    diff = tm - mean_tot[:, np.newaxis]
    var = (tw * (tv + diff**2)).sum(axis=1)/wsum
    sigma_tot = np.sqrt(var)

    print('mean tot:', mean_tot)
    print('mean tot scaled:', mean_tot/sigma_tot)
    print('sigma tot:', sigma_tot)

    for igauss in range(ngauss):

        weights = struct['%s_weights' % type][:, igauss]
        s_means = (struct['%s_means' % type][:, igauss] - mean_tot)/sigma_tot
        s_sigmas = struct['%s_sigmas' % type][:, igauss]/sigma_tot

        wmean = weights.mean()
        print(type, igauss, 'weight mean: %.17g' % wmean)
        tab[0, 0].axhline(wmean, color='black')
        tab[0, 0].curve(centers, weights, marker='o', markersize=2)

        s_mean_mean = s_means[:5].mean()
        print(type, igauss, 'mean mean (zerod): %.17g' % s_mean_mean)
        tab[0, 1].axhline(s_mean_mean, color='black')
        tab[0, 1].curve(centers, s_means, marker='o', markersize=1.5)

        s_sigma_mean = s_sigmas.mean()
        print(type, igauss, 'sigma mean: %.17g' % s_sigma_mean)
        tab[1, 0].axhline(s_sigma_mean, color='black')
        tab[1, 0].curve(centers, s_sigmas, marker='o', markersize=1.5)

    if show:
        tab.show()

    if output is not None:
        print('writing:', output)
        tab.savefig(output, dpi=100)

    return tab


def plot_all(*, struct, type, show=False, output=None):

    if type == 'star':
        label = 'stars'
    else:
        label = 'galaxies'

    tab = hickory.Table(
        figsize=(11, 11*0.618),
        nrows=2,
        ncols=2,
    )

    tab[1, 1].axis('off')

    tab[0, 0].set(
        xlabel='r mag',
        ylabel='weight',
    )
    tab[0, 1].set(
        xlabel='r mag',
        ylabel='mean',
    )
    tab[1, 0].set(
        xlabel='r mag',
        ylabel=r'$\sigma$',
    )

    tab[1, 1].ntext(
        0.5, 0.5, label,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16,
    )

    centers = struct['rmag_centers']
    ngauss = struct['%s_weights' % type].shape[1]

    for igauss in range(ngauss):

        weights = struct['%s_weights' % type][:, igauss]
        means = struct['%s_means' % type][:, igauss]
        sigmas = struct['%s_sigmas' % type][:, igauss]

        wmean = weights.mean()
        print(type, igauss, wmean)
        tab[0, 0].axhline(wmean, color='black')
        tab[0, 0].curve(centers, weights, marker='o', markersize=2)

        if type == 'gal':
            poly = np.poly1d(np.polyfit(centers, means, 2))
            print(poly)
            tab[0, 1].curve(centers, poly(centers), color='black')
        else:
            pass
            """
            from .fitting import fit_exp, exp_func

            if igauss == 0:
                guess = [-3.0e-9, 16, 0.5]
            elif igauss == 1:
                guess = [-4.5e-10, 17, 0.5]
            else:
                guess = [+7.5e-8, 11, 1.1]

            res = fit_exp(centers, means, guess)
            # assert res['flags'] == 0
            print('star', igauss, 'mean flags:', res['flags'])
            print_pars(
                res['pars'], front='star %d mean exp pars: ' % igauss,
                fmt='%.6g',
            )

            if res['flags'] == 0:
                p = exp_func(res['pars'], centers)
                pcolor = 'black'
            else:
                p = exp_func(guess, centers)
                pcolor = 'red'

            tab[0, 1].curve(
                centers,
                # exp_func(res['pars'], centers),
                p,
                color=pcolor,
            )
            # tab[0, 1].set(ylim=[-0.0012, 0.007])
            # tab[0, 1].set(ylim=[-0.0005, 0.0005])
            """

        tab[0, 1].curve(centers, means, marker='o', markersize=1.5)

        if type == 'star':
            from .fitting import fit_exp_pedestal, exp_func_pedestal

            if igauss == 0:
                guess = [6.4e-09, 12.2, 0.96, 6.5e-05]
            elif igauss == 1:
                # guess = [1.5e-08, 10, 1.3, 1.5e-05]
                guess = [6.4e-09, 11.5, 0.96, 0.0001]
            else:
                # guess = [6.5e-08, 9.6, 1.3, 0.0002]
                guess = [2e-07, 11, 1.3, 0.00013]

            from . import interp
            sigmas = interp.smooth_data_hann3(sigmas.astype('f8'))
            res = fit_exp_pedestal(centers, sigmas, guess)

            # assert res['flags'] == 0
            print('star', igauss, 'sigma flags:', res['flags'])
            print_pars(
                res['pars'], front='star %d sigma exp pars: ' % igauss,
                fmt='%.6g',
            )

            # if res['flags'] == 0:
            #     p = exp_func_pedestal(res['pars'], centers)
            #     pcolor = 'black'
            # else:
            #     p = exp_func_pedestal(guess, centers)
            #     pcolor = 'red'
            p = exp_func_pedestal(guess, centers)
            pcolor = 'red'

            tab[1, 0].curve(
                centers,
                p,
                color=pcolor,
            )
            # tab[0, 1].set(ylim=[-0.0012, 0.007])
            # tab[1, 0].set(ylim=[0, 0.003])
            tab[1, 0].set_yscale('log')

        tab[1, 0].curve(centers, sigmas, marker='o', markersize=1.5)

    if show:
        tab.show()

    if output is not None:
        print('writing:', output)
        tab.savefig(output, dpi=100)

    return tab


def get_output_file(*, infile, extra=None, ext='fits'):

    new_end = ['', 'priors']
    if extra is not None:
        new_end += [extra]

    new_end = '-'.join(new_end)+'.'+ext
    outfile = infile.replace(
        '.fits',
        new_end,
    )
    assert outfile != infile

    return outfile


def fit_priors(*, seed, file, rmag_index, show=False):

    rng = np.random.RandomState(seed)

    star_ngauss = 3
    gal_ngauss = 2

    extra0 = 'star%d-gal%d' % (star_ngauss, gal_ngauss)

    pdf_file = get_output_file(
        infile=file,
        extra=extra0+'-plots',
        ext='pdf',
    )

    print('writing plots to:', pdf_file)
    pdf = PdfPages(pdf_file)

    data = fitsio.read(file)

    edges = [
        (19.0, 20.0),
        # (19.0, 19.5),
        # (19.5, 20.0),
        (20, 20.5),
        (20.5, 21.0),
        (21, 21.5),
        (21.5, 22.0),
        (22, 22.5),
        (22.5, 23),
        (23, 23.5),
        (23.5, 24.0),
        (24, 24.5),
        # (24.5, 25),
    ]

    # power = 0.5
    # off = 0.015
    struct = get_struct(
        size=len(edges), star_ngauss=star_ngauss, gal_ngauss=gal_ngauss,
    )
    for i in range(len(edges)):
        rmagmin, rmagmax = edges[i]

        wstar = select_stars(
            data=data,
            rmagmin=rmagmin, rmagmax=rmagmax,
            rmag_index=rmag_index,
        )
        wgal = select_gals(
            data=data,
            rmagmin=rmagmin, rmagmax=rmagmax,
            rmag_index=rmag_index,
        )
        star_conc = data['conc'][wstar]
        gal_conc = data['conc'][wgal]

        # star_conc = (star_conc + off)**power - off**power
        # gal_conc = (gal_conc + off)**power - off**power

        print('nstars:', wstar.size)
        print('ngals:', wgal.size)

        label = r'[%.2f, %.2f]' % (rmagmin, rmagmax)

        # star_gmm, star_plt = fit_gmm(
        star_gmm, star_plt = fit_stars_constrained(
            rng=rng,
            data=star_conc,
            ngauss=star_ngauss,
            show=show,
            label='stars rmag: %s' % label,
        )
        pdf.savefig(figure=star_plt)
        # gal_gmm, gal_plt = fit_gmm(
        gal_gmm, gal_plt = fit_gals_constrained(
            rng=rng,
            data=gal_conc,
            ngauss=gal_ngauss,
            show=show,
            label='gals rmag: %s' % label,
        )
        pdf.savefig(figure=gal_plt)

        tstruct = pack_struct(
            rmagmin=rmagmin,
            rmagmax=rmagmax,
            star_gmm=star_gmm, gal_gmm=gal_gmm,
        )
        struct[i] = tstruct

    star_plt = plot_all(struct=struct, type='star', show=show)
    gal_plt = plot_all(struct=struct, type='gal', show=show)
    star_scaled_plt = plot_all_scaled(struct=struct, type='star', show=show)
    gal_scaled_plt = plot_all_scaled(struct=struct, type='gal', show=show)

    pdf.savefig(figure=star_plt)
    pdf.savefig(figure=star_scaled_plt)
    pdf.savefig(figure=gal_plt)
    pdf.savefig(figure=gal_scaled_plt)
    print('closing:', pdf_file)
    pdf.close()

    outfile = get_output_file(infile=file, extra=extra0)
    print('writing:', outfile)
    fitsio.write(outfile, struct, clobber=True)
