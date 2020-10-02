"""
TODO
    - make purity plots automatically?
    - write out new file with probabilities?  Depends on interpolation so might
      want to do in separate run of a script
"""
import numpy as np
from esutil.numpy_util import between
from . import staramp
from desclass.cem import (
    gmix_get_weight,
    gmix_get_mean,
    gmix_get_sigma,
    gmix_print,
)

from . import star_em
# from . import bf
from .star_em import make_constraints
from .interp import smooth_data_hann3, interp_gp
from .fitting import exp_func_pedestal

from matplotlib.backends.backend_pdf import PdfPages


STAR_SIGMA_PARS = np.array([1.56451003e-04, 2.07155232e+01, 1.24008171e+00, 1.26820679e-04])  # noqa


def star_sigma_vs_rmag(*, rmag):
    return exp_func_pedestal(STAR_SIGMA_PARS, rmag,)


def star_mean_vs_rmag(*, rmag):
    # return bf.predict(rmag=rmag, amp=-1.25e-06)
    ply = np.poly1d([-1.93076955e-05,  3.64168874e-04])
    return ply(rmag)


def get_star_constraints(*, weight, rmag):

    ww = 0.01
    # ww = 0.10
    sw = 0.5
    # mw = 1.0e-5
    mw = 1.0e-6
    mean = star_mean_vs_rmag(rmag=rmag)
    sigma = star_sigma_vs_rmag(rmag=rmag)
    constraints = make_constraints(
        weight_low=(1-ww/2)*weight,
        weight_high=(1+ww/2)*weight,
        # mean_low=-7.5e-5,
        # mean_low=-1.0e-4,
        # mean_high=5.0e-5,
        # mean_high=0.0,
        mean_low=mean-mw,
        mean_high=mean+mw,
        sigma_low=(1-sw/2)*sigma,
        sigma_high=(1+sw/2)*sigma,
    )
    return constraints


GAL1_MEAN_POLY = np.poly1d([-4.33834694e-05, 9.92808770e-04, 5.12932203e-03])
GAL2_MEAN_POLY = np.poly1d([-0.00024022,  0.0092886, -0.07650446])


def gal1_mean_vs_rmag(*, rmag):
    return GAL1_MEAN_POLY(rmag)


def gal2_mean_vs_rmag(*, rmag):
    return GAL2_MEAN_POLY(rmag)


# very rough and need to not extrapolate at low end
GAL1_SIGMA_POLY = np.poly1d([7.08930960e-06, -3.27674122e-05, -3.14135302e-04])
GAL2_SIGMA_POLY = np.poly1d([0.00012233, -0.00501155,  0.05422102])


def gal1_sigma_vs_rmag(*, rmag):
    if rmag < 19.5:
        rmag = 19.5
    return GAL1_SIGMA_POLY(rmag)


def gal2_sigma_vs_rmag(*, rmag):
    if rmag < 19.5:
        rmag = 19.5
    return GAL2_SIGMA_POLY(rmag)


def get_gal_constraints(*, weight, rmag):

    ww = 0.01
    gal1_weight = 0.57 * weight
    gal2_weight = 0.43 * weight

    mw = 0.10
    gal1_mean = gal1_mean_vs_rmag(rmag=rmag)
    gal2_mean = gal2_mean_vs_rmag(rmag=rmag)

    if rmag < 19.5:
        gal1_sigma_low = 0.0025
        gal1_sigma_high = 0.01
        gal2_sigma_low = 0.0025
        gal2_sigma_high = 0.01
    else:
        gal1_sigma = gal1_sigma_vs_rmag(rmag=rmag)
        gal2_sigma = gal2_sigma_vs_rmag(rmag=rmag)
        gal1_sigma_low = 0.7*gal1_sigma
        gal1_sigma_high = 1.3*gal1_sigma
        gal2_sigma_low = 0.7*gal2_sigma
        gal2_sigma_high = 1.3*gal2_sigma

    constraints = np.hstack(
        [
            make_constraints(
                weight_low=(1-ww/2)*gal1_weight,
                weight_high=(1+ww/2)*gal1_weight,
                mean_low=(1-mw/2)*gal1_mean,
                mean_high=(1+mw/2)*gal1_mean,
                sigma_low=gal1_sigma_low,
                sigma_high=gal1_sigma_high,
            ),
            make_constraints(
                weight_low=(1-ww/2)*gal2_weight,
                weight_high=(1+ww/2)*gal2_weight,
                mean_low=(1-mw/2)*gal2_mean,
                mean_high=(1+mw/2)*gal2_mean,
                sigma_low=gal2_sigma_low,
                sigma_high=gal2_sigma_high,
            )]
    )
    return constraints


def replace_ext(fname, old_ext, new_ext):
    new_fname = fname.replace(old_ext, new_ext)
    assert new_fname != fname
    return new_fname


def make_output(num):
    gdt = [
        ('weight', 'f8'),
        ('mean', 'f8'),
        ('sigma', 'f8'),
        ('icovar', 'f8'),
        ('norm', 'f8'),
        ('num', 'f8'),
    ]

    dt = [
        ('rmag', 'f8'),
        ('rmagmin', 'f8'),
        ('rmagmax', 'f8'),
        ('gmix', gdt, 5),
    ]

    return np.zeros(num, dtype=dt)


def plot_star_fits_vs_rmag(data, dofits=False, show=False):
    """
    make a plot of gaussian mixture parameters vs the central
    magnitude of the bin
    """
    import hickory

    tab = hickory.Table(
        nrows=2,
        ncols=2,
    )

    xlim = (15.5, 25.5)
    tab.suptitle('stars', fontsize=15)
    tab[0, 0].set(
        xlabel='r mag',
        ylabel='weight',
        xlim=xlim,
    )
    tab[0, 1].set(
        xlabel='r mag',
        ylabel='mean',
        xlim=xlim,
        ylim=(-0.00012, 0.00008),
    )
    tab[1, 0].set(
        xlabel='r mag',
        ylabel=r'$\sigma$',
        xlim=xlim,
    )
    tab[1, 1].axis('off')
    tab[1, 1].ntext(0.5, 0.5, 'stars',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=16)

    centers = data['rmag']
    star_gmix = data['gmix'][:, :3]

    weights = np.array([gmix_get_weight(gm) for gm in star_gmix])
    means = np.array([gmix_get_mean(gm) for gm in star_gmix])
    sigmas = np.array([gmix_get_sigma(gm) for gm in star_gmix])

    tab[0, 0].plot(centers, weights, marker='o', markersize=2)
    # tab[0, 1].plot(centers, relweights, marker='o', markersize=2)

    tab[0, 1].plot(centers, means, marker='o', markersize=2)
    tab[1, 0].plot(centers, sigmas, marker='o', markersize=2)

    # interpolation
    xinterp = np.linspace(centers[0], centers[-1], 1000)

    ystd = (smooth_data_hann3(weights) - weights).std()
    yinterp, ysigma = interp_gp(centers, weights, ystd, xinterp)
    tab[0, 0].curve(xinterp, yinterp, linestyle='solid')

    ystd = (smooth_data_hann3(means) - means).std()
    yinterp, ysigma = interp_gp(centers, means, ystd, xinterp)
    tab[0, 1].curve(xinterp, yinterp, linestyle='solid')

    ystd = (smooth_data_hann3(sigmas) - sigmas).std()
    ysend = smooth_data_hann3(sigmas)

    yinterp, ysigma = interp_gp(centers, ysend, ystd, xinterp)
    tab[1, 0].curve(xinterp, yinterp, linestyle='solid')

    if show:
        tab.show()

    return tab


def plot_gal_fits_vs_rmag(data, dofits=False, show=False):
    """
    make a plot of gaussian mixture parameters vs the central
    magnitude of the bin
    """
    import hickory

    tab = hickory.Table(
        nrows=2,
        ncols=2,
    )

    xlim = (15.5, 25.5)
    tab.suptitle('galaxies', fontsize=15)
    tab[0, 0].set(
        xlabel='r mag',
        ylabel='weight',
        xlim=xlim
    )
    # tab[0, 1].set(
    #     xlabel='r mag',
    #     ylabel='relweight',
    #     xlim=xlim,
    # )
    tab[0, 1].set(
        xlabel='r mag',
        ylabel='mean',
        xlim=xlim,
    )

    tab[1, 0].set(
        xlabel='r mag',
        ylabel=r'$\sigma$',
        xlim=xlim,
    )
    tab[1, 1].axis('off')
    tab[1, 1].ntext(0.5, 0.5, 'galaxies',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=16)

    centers = data['rmag']
    gmixes = data['gmix'][:, 3:]

    # wsums = gmixes['weight'].sum(axis=1)

    ngauss = gmixes['weight'].shape[1]

    colors = ['#1f77b4', '#ff7f0e']
    for igauss in range(ngauss):
        color = colors[igauss]

        weights = gmixes['weight'][:, igauss]
        means = gmixes['mean'][:, igauss]
        sigmas = gmixes['sigma'][:, igauss]

        # relweights = weights/wsums

        tab[0, 0].plot(centers, weights, marker='o', markersize=2,
                       color=color)
        # tab[0, 1].plot(centers, relweights, marker='o', markersize=2,
        #                color=color)

        tab[0, 1].plot(centers, means, marker='o', markersize=2, color=color)
        tab[1, 0].plot(centers, sigmas, marker='o', markersize=2, color=color)

        # interpolation
        xinterp = np.linspace(centers[0], centers[-1], 1000)

        ystd = (smooth_data_hann3(weights) - weights).std()
        yinterp, ysigma = interp_gp(centers, weights, ystd, xinterp)
        tab[0, 0].curve(xinterp, yinterp, linestyle='solid',
                        color=color)

        ystd = (smooth_data_hann3(means) - means).std()
        yinterp, ysigma = interp_gp(centers, means, ystd, xinterp)
        tab[0, 1].curve(xinterp, yinterp, linestyle='solid',
                        color=color)

        ystd = (smooth_data_hann3(sigmas) - sigmas).std()
        ysend = smooth_data_hann3(sigmas)

        yinterp, ysigma = interp_gp(centers, ysend, ystd, xinterp)
        tab[1, 0].curve(xinterp, yinterp, linestyle='solid',
                        color=color)

    if show:
        tab.show()

    return tab


def fit_conc_pdf(
    *,
    data, rmag_index, seed, output,
    show=False,
):

    rng = np.random.RandomState(seed)

    print('will write to:', output)
    pdf_file = replace_ext(output, '.npy', '-plots.pdf')
    print('writing to pdf file:', pdf_file)
    pdf = PdfPages(pdf_file)

    data = data[data['flags'] == 0]

    edges = [
        (16.5, 17.0),
        (17.0, 17.5),
        (17.5, 18.0),
        (18.0, 18.5),

        (18.5, 19.0),
        (19, 19.5),
        (19.5, 20.0),
        (20, 20.5),
        (20.5, 21.0),
        (21, 21.5),
        (21.5, 22.0),
        (22, 22.5),
        (22.5, 23),
        (23, 23.5),
        (23.5, 24.0),
        (24, 24.5),
    ]

    rmag_centers = np.array(
        [0.5*(rmagmin + rmagmax) for rmagmin, rmagmax in edges]
    )

    outdata = make_output(len(edges))

    # initial estimate of N(mag) for stars
    initial_amp, initial_amp_err, plt = staramp.get_amp(
        rmag=data['psf_mag'][:, rmag_index],
        conc=data['conc'],
        show=show,
        get_plot=True,
    )
    pdf.savefig(figure=plt)

    init_nstar, init_nstar_err = staramp.predict(
        rmag=rmag_centers,
        amp=initial_amp,
        amp_err=initial_amp_err,
    )

    for i in range(len(edges)):
        rmagmin, rmagmax = edges[i]
        rmag = 0.5*(rmagmin + rmagmax)

        label = r'$%.2f < r < %.2f$' % (rmagmin, rmagmax)
        print('-'*70)
        print(label)

        minconc, maxconc = -0.01, 0.025
        w, = np.where(
            between(data['psf_mag'][:, rmag_index], rmagmin, rmagmax) &
            between(data['conc'], minconc, maxconc)
        )
        print('number in bin:', w.size)

        nstar_predicted = init_nstar[i]
        ngal_predicted = w.size - nstar_predicted
        if ngal_predicted < 3:
            ngal_predicted = 3

        star_weight = nstar_predicted/w.size
        gal_weight = ngal_predicted/w.size

        star_constraints = get_star_constraints(weight=star_weight, rmag=rmag)
        gal_constraints = get_gal_constraints(weight=gal_weight, rmag=rmag)
        fitter = star_em.StarEMFitter(
            data=data['conc'][w],
            star_constraints=star_constraints,
            gal_constraints=gal_constraints,
            rng=rng,
        )
        fitter.go()

        res = fitter.result
        print('final')
        gmix_print(fitter.gmix)
        print(res)
        assert res['flags'] == 0, 'failed to converge'

        plt = fitter.plot3(label=label, show=show)
        pdf.savefig(figure=plt)

        outdata['rmagmin'][i] = rmagmin
        outdata['rmagmax'][i] = rmagmax
        outdata['rmag'][i] = rmag
        outdata['gmix'][i] = fitter.gmix

    plt = plot_star_fits_vs_rmag(outdata, show=show)
    pdf.savefig(plt)
    plt = plot_gal_fits_vs_rmag(outdata, show=show)
    pdf.savefig(plt)

    print('closing pdf file:', pdf_file)
    pdf.close()
    print('writing:', output)
    np.save(output, outdata)
