import numpy as np
import fitsio
import esutil as eu
from esutil.numpy_util import between
import hickory
import ngmix
# import scipy.optimize
from ngmix.fitting import run_leastsq
# from ngmix import print_pars
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

    if show or output is not None:
        std = data.std()
        binsize = std/7

        gmm.plot(
            data=data, binsize=binsize, title=label,
            legend=True,
            show=show,
            file=output,
        )

    return gmm


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
    minconc, maxconc = [mn - 4*std, mn + 6*std]
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
        ('star_covars', 'f8', star_ngauss),

        ('gal_weights', 'f8', gal_ngauss),
        ('gal_means', 'f8', gal_ngauss),
        ('gal_covars', 'f8', gal_ngauss),
    ]
    return np.zeros(size, dtype=dt)


def pack_struct(*, rmagmin, rmagmax, star_gmm, gal_gmm):
    struct = get_struct(
        size=1,
        star_ngauss=star_gmm.weights.size,
        gal_ngauss=gal_gmm.weights.size,
    )

    struct['rmagmin'] = rmagmin
    struct['rmagmax'] = rmagmax
    struct['rmag_centers'] = 0.5*(rmagmax + rmagmin)

    s = (-star_gmm.weights.ravel()).argsort()
    struct['star_weights'][0] = star_gmm.weights.ravel()[s]
    struct['star_means'][0] = star_gmm.means.ravel()[s]
    struct['star_covars'][0] = star_gmm.covars.ravel()[s]

    s = (-gal_gmm.weights.ravel()).argsort()
    struct['gal_weights'][0] = gal_gmm.weights.ravel()[s]
    struct['gal_means'][0] = gal_gmm.means.ravel()[s]
    struct['gal_covars'][0] = gal_gmm.covars.ravel()[s]

    return struct


def expfunc(pars, x):
    amp = pars[0]
    off = pars[1]
    sigma = pars[2]

    model = np.zeros(x.size)
    w, = np.where(x > off)
    if w.size > 0:
        arg = ((x[w] - off)/sigma)**2
        model[w] = amp * np.exp(arg)

    return model


def fitexp(x, y, guess):
    # assume quadratic

    def loss(pars):
        model = expfunc(pars, x)
        return (model - y)

    return run_leastsq(
        loss,
        np.array(guess),
        0,
    )


def plot_all(*, struct, type, show=False, output=None):

    if type == 'star':
        label = 'stars'
    else:
        label = 'galaxies'

    tab = hickory.Table(
        nrows=2,
        ncols=2,
    )

    tab[1, 1].axis('off')

    tab[0, 0].ntext(0.1, 0.5, label, verticalalignment='center')
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
        ylabel=r'$\sigma^2$',
    )

    centers = struct['rmag_centers']
    ngauss = struct['%s_weights' % type].shape[1]

    for igauss in range(ngauss):

        weights = struct['%s_weights' % type][:, igauss]
        means = struct['%s_means' % type][:, igauss]
        covars = struct['%s_covars' % type][:, igauss]

        wmean = weights.mean()
        print(type, igauss, wmean)
        tab[0, 0].axhline(wmean, color='black')
        tab[0, 0].curve(centers, weights, marker='o', markersize=2)

        if type == 'gal':
            poly = np.poly1d(np.polyfit(centers, means, 2))
            print(poly)
            tab[0, 1].curve(centers, poly(centers), color='black')
        else:
            res = fitexp(centers, means, [0.0005, 18, 4])

            tab[0, 1].curve(centers, expfunc(res['pars'], centers),
                            color='black', linestyle='solid')
            print('means pars:', res['pars'])

        tab[0, 1].curve(centers, means, marker='o', markersize=1.5)

        if type == 'star':
            res = fitexp(centers, covars, [0.3e-6, 18, 3])

            if res['flags'] == 0:
                tab[1, 0].curve(centers, expfunc(res['pars'], centers),
                                color='black', linestyle='solid')

            print('covars pars:', res['pars'])

        tab[1, 0].curve(centers, covars, marker='o', markersize=1.5)

    if show:
        tab.show()
    if output is not None:
        tab.savefig(output, dpi=100)


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

    data = fitsio.read(file)

    star_ngauss = 2
    gal_ngauss = 2

    # off = 0.007
    # power = 0.5

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
        star_pdf_file = get_output_file(
            infile=file, extra='stars-%.2f-%.2f' % (rmagmin, rmagmax),
            ext='pdf',
        )
        gal_pdf_file = get_output_file(
            infile=file, extra='gals-%.2f-%.2f' % (rmagmin, rmagmax),
            ext='pdf',
        )

        star_gmm = fit_gmm(
            rng=rng,
            data=star_conc,
            ngauss=star_ngauss,
            show=show,
            output=star_pdf_file,
            label='stars rmag: %s' % label,
        )
        gal_gmm = fit_gmm(
            rng=rng,
            data=gal_conc,
            ngauss=gal_ngauss,
            show=show,
            output=gal_pdf_file,
            label='gals rmag: %s' % label,
        )

        tstruct = pack_struct(
            rmagmin=rmagmin,
            rmagmax=rmagmax,
            star_gmm=star_gmm, gal_gmm=gal_gmm,
        )
        struct[i] = tstruct

    star_pdf_file = get_output_file(
        infile=file, extra='star-trends', ext='pdf',
    )
    gal_pdf_file = get_output_file(
        infile=file, extra='gal-trends', ext='pdf',
    )
    plot_all(struct=struct, type='star', output=star_pdf_file)
    plot_all(struct=struct, type='gal', output=gal_pdf_file)

    outfile = get_output_file(infile=file)
    print('writing:', outfile)
    fitsio.write(outfile, struct, clobber=True)
