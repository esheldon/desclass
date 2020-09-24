import numpy as np
import fitsio
import esutil as eu
from esutil.numpy_util import between
import hickory
# import scipy.optimize
# from ngmix.fitting import run_leastsq
from ngmix import print_pars
# from glob import glob
from . import fithist
# from matplotlib.backends.backend_pdf import PdfPages


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


def get_struct(size=1):
    dt = [
        ('rmagmin', 'f8'),
        ('rmagmax', 'f8'),
        ('rmag_centers', 'f8'),

        ('star_a', 'f8'),
        ('star_loc', 'f8'),
        ('star_scale', 'f8'),

        ('gal1_weight', 'f8'),
        ('gal1_mean', 'f8'),
        ('gal1_sigma', 'f8'),

        ('gal2_weight', 'f8'),
        ('gal2_mean', 'f8'),
        ('gal2_sigma', 'f8'),
    ]
    return np.zeros(size, dtype=dt)


def pack_struct(*, rmagmin, rmagmax, star_pars, gal_pars):
    struct = get_struct()

    struct['rmagmin'] = rmagmin
    struct['rmagmax'] = rmagmax
    struct['rmag_centers'] = 0.5*(rmagmax + rmagmin)

    struct['star_a'][0] = star_pars[1]
    struct['star_loc'][0] = star_pars[2]
    struct['star_scale'][0] = star_pars[3]

    gal1_amp = gal_pars[0]
    gal2_amp = gal_pars[3]

    amp_sum = gal1_amp + gal2_amp
    struct['gal1_weight'] = gal1_amp/amp_sum
    struct['gal1_mean'] = gal_pars[1]
    struct['gal1_sigma'] = gal_pars[2]

    struct['gal2_weight'] = gal2_amp/amp_sum
    struct['gal2_mean'] = gal_pars[4]
    struct['gal2_sigma'] = gal_pars[5]

    return struct


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
        ylabel=r'$\sigma$',
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
            from .fitting import fit_exp, exp_func

            if igauss == 0:
                guess = [-1.0e-6, 16, 1]
            else:
                guess = [+1.0e-6, 16, 1]
            res = fit_exp(centers, means, guess)
            assert res['flags'] == 0
            print_pars(
                res['pars'], front='star mean exp pars: ',
                fmt='%.6g',
            )
            tab[0, 1].curve(centers, exp_func(res['pars'], centers),
                            color='black')

        tab[0, 1].curve(centers, means, marker='o', markersize=1.5)
        tab[1, 0].curve(centers, sigmas, marker='o', markersize=1.5)

    if show:
        tab.show()

    if output is not None:
        print('writing:', output)
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


def fit_stars(*, rng, conc, title, ntry=100):
    print('fitting stars')
    mn, std = eu.stat.sigma_clip(conc)
    binsize = std/5

    nsig = 4
    minconc, maxconc = mn - nsig*std, mn + nsig*std
    hd = eu.stat.histogram(
        conc,
        min=minconc,
        max=maxconc,
        binsize=binsize,
        more=True,
    )

    amp = conc.size * binsize
    prior = fithist.StarPrior(
        rng=rng,
        amp=amp,
        amp_sigma=amp*0.01,
        a=3,
        a_sigma=1,
        loc=mn,
        loc_sigma=std*0.3,
        scale=std/2,
        scale_sigma=std/2*0.3,
    )

    fitter = fithist.Fitter(
        x=hd['center'],
        y=hd['hist'],
        yerr=np.sqrt(hd['hist']),
        prior=prior,
        model_class=fithist.StarModel,
    )

    for i in range(ntry):
        guess = prior.sample()
        print_pars(guess, front='    guess: ')
        fitter.go(guess)
        res = fitter.result
        if res['flags'] == 0:
            break

    assert res['flags'] == 0

    print_pars(res['pars'], front='    pars: ')
    print('nfev:', res['nfev'])
    print('flags:', res['flags'])

    plt = hickory.Plot(
        # figsize=(10, 6.5),
        xlabel='concentration',
        legend=True,
        title=title,
    )
    plt.bar(
        hd['center'], hd['hist'], binsize, alpha=0.5,
        color='#a6a6a6',
        label='data',
    )

    plotx = np.linspace(hd['center'][0], hd['center'][-1], 1000)

    starmod = fithist.StarModel(res['pars'])
    starmod.plot(x=plotx, plt=plt)
    plt.show()

    return res['pars']


def replace_ext(fname, old_ext, new_ext):
    new_fname = fname.replace(old_ext, new_ext)
    assert new_fname != fname
    return new_fname


def fit_priors(*, seed, file, rmag_index, show=False):

    outfile = get_output_file(infile=file)
    pdf_file = replace_ext(outfile, '.fits', '-plots.pdf')
    print('writing to pdf file:', pdf_file)
    # pdf = PdfPages(pdf_file)

    rng = np.random.RandomState(seed)

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
    # edges = [(24.5, 25)]
    # edges = [(20.5, 21)]
    # edges = [(24.0, 24.1)]

    # power = 0.5
    # off = 0.015
    struct = get_struct(len(edges))
    for i in range(len(edges)):
        rmagmin, rmagmax = edges[i]
        label = r'$%.2f < r < %.2f$' % (rmagmin, rmagmax)

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
        # gal_conc = data['conc'][wgal]

        print('nstars:', wstar.size)
        print('ngals:', wgal.size)

        # label = r'[%.2f, %.2f]' % (rmagmin, rmagmax)

        star_pars = fit_stars(rng=rng, conc=star_conc, title=label)
        gal_pars = np.ones(6)

        tstruct = pack_struct(
            rmagmin=rmagmin,
            rmagmax=rmagmax,
            star_pars=star_pars,
            gal_pars=gal_pars,
        )
        struct[i] = tstruct

    # plot_all(struct=struct, type='star', output=star_pdf_file, show=show)
    # plot_all(struct=struct, type='gal', output=gal_pdf_file, show=show)
    #
    # print('writing:', outfile)
    # fitsio.write(outfile, struct, clobber=True)
