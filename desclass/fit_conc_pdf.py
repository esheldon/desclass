import numpy as np
import esutil as eu
from esutil.numpy_util import between
import fitsio
# import hickory
# import scipy.optimize
# from ngmix import print_pars
from . import staramp
# from . import galamp
from . import cem

from matplotlib.backends.backend_pdf import PdfPages


class GaussianPrior(object):
    def __init__(self, *, mean, sigma, bounds, rng):
        self.mean = mean
        self.sigma = sigma
        self.rng = rng
        self.bounds = bounds

    def get_fdiff(self, value):
        return (value - self.mean)/self.sigma

    def sample(self, n=None, sigma_factor=1):
        if n is not None:
            scalar = False
        else:
            n = 1
            scalar = True

        rand = np.zeros(n)

        ngood = 0
        nleft = n
        while nleft > 0:
            tmp = self.rng.normal(
                size=n,
                loc=self.mean,
                scale=self.sigma*sigma_factor,
            )
            w, = np.where(between(tmp, self.bounds[0], self.bounds[1]))
            if w.size > 0:
                rand[ngood:ngood + w.size] = tmp
                ngood += w.size
                nleft -= w.size

        if scalar:
            rand = rand[0]

        return rand

    def __repr__(self):
        return 'mean: %g sigma: %g' % (self.mean, self.sigma)


def _gal_mean2_vs_rmag(*, rmag):
    return -4.729e-05 * rmag**2 + 0.001028 * rmag + 0.015


def _gal_mean1_vs_rmag(*, rmag):
    return -4.729e-05 * rmag**2 + 0.001028 * rmag + 0.006156


GAL_WT_ERF_PARS = np.array([0.61744558, 0.93249301, 19.90167776, 0.81298793])


def _gal_relweight_vs_rmag(*, rmag):
    from .fitting import erf_func
    return erf_func(GAL_WT_ERF_PARS, rmag)


def get_gal_priors(*, rng, data, rmag, ngal, ngal_err, ngauss, training=False):
    """
    get star priors.  The gaussian pdfs are used for generating
    guesses with bounds, only the bounds themselves are used in
    the EM algorithm
    """
    if ngauss == 1:
        frac1 = 1
    elif ngauss == 2:
        if training:
            frac1 = 0.6
            num_bwidth = 0.9
        else:
            frac1 = _gal_relweight_vs_rmag(rmag=rmag)
            num_bwidth = 0.1
        frac2 = 1 - frac1
    else:
        raise ValueError('ngauss should be 1 or 2')

    sigma_sigma_frac = 0.3

    # number in each gaussian
    num1 = frac1*ngal
    num1_err = frac1*ngal_err
    num1_prior = GaussianPrior(
        mean=num1,
        sigma=num1_err,
        # bounds=[0.1*num1, 2*num1],
        bounds=[(1-num_bwidth)*num1, (1+num_bwidth)*num1],
        rng=rng,
    )
    if ngauss == 2:
        num2 = frac2*ngal
        num2_err = frac2*ngal_err
        num2_prior = GaussianPrior(
            mean=num2,
            sigma=num2_err,
            # bounds=[0.1*num2, 2*num2],
            bounds=[(1-num_bwidth)*num2, (1+num_bwidth)*num2],
            rng=rng,
        )

    # mean of gaussians
    if ngauss == 1:
        # mean1 = 0.01
        tmean1 = _gal_mean1_vs_rmag(rmag=rmag)
        tmean2 = _gal_mean2_vs_rmag(rmag=rmag)
        print('tmean1:', tmean1, 'tmean2:', tmean2)

        mean1 = (tmean1*0.6 + tmean2*0.4)
        print('mean1:', mean1)
        mean_sigma_frac = 0.1
        mean_bounds1 = [0.0025, 0.025]
    else:
        mean1 = _gal_mean1_vs_rmag(rmag=rmag)
        mean2 = _gal_mean2_vs_rmag(rmag=rmag)
        mean_sigma_frac = 0.05

        mean_bounds1 = [
            mean1 - mean_sigma_frac*abs(mean1),
            mean1 + mean_sigma_frac*abs(mean1),
        ]
        if rmag < data['rmag_centers'][0]:
            mean_bounds2 = [
                mean2 - mean_sigma_frac*abs(mean2),
                mean2 + mean_sigma_frac*abs(mean2),
            ]
        else:
            mean_bounds2 = [0.0025, 0.025]

    mean1_prior = GaussianPrior(
        mean=mean1,
        sigma=abs(mean_sigma_frac*mean1),
        bounds=mean_bounds1,
        # bounds=[0.0025, 0.02],
        rng=rng,
    )

    if ngauss == 2:
        mean2_prior = GaussianPrior(
            mean=mean2,
            sigma=abs(mean_sigma_frac*mean2),
            # bounds=[0.007, 0.02],
            # bounds=[0.0025, 0.025],
            bounds=mean_bounds2,
            rng=rng,
        )

    # no extrapolation for sigma, just to linear interpolation
    # and use endpoints for those outside

    sigma1 = np.interp(rmag, data['rmag_centers'], data['gal_sigmas'][:, 0])
    sigma2 = np.interp(rmag, data['rmag_centers'], data['gal_sigmas'][:, 1])
    if rmag < data['rmag_centers'][0]:
        sigma1_bounds = [0.003, 0.007]
        sigma2_bounds = [0.003, 0.007]
    else:
        sigma1_bounds = [0.1*sigma1, 2*sigma1]
        sigma2_bounds = [0.1*sigma1, 2*sigma1]

    sigma1_prior = GaussianPrior(
        mean=sigma1,
        sigma=abs(sigma_sigma_frac*sigma1),
        bounds=sigma1_bounds,
        rng=rng,
    )

    if ngauss == 2:
        sigma2_prior = GaussianPrior(
            mean=sigma2,
            sigma=abs(sigma_sigma_frac*sigma2),
            bounds=sigma2_bounds,
            rng=rng,
        )

    print()
    print('gal mean1_prior:', mean1_prior)
    print('gal sigma1_prior:', sigma1_prior)
    print('gal num1_prior:', num1_prior)

    if ngauss == 2:
        print('gal mean2_prior:', mean2_prior)
        print('gal sigma2_prior:', sigma2_prior)
        print('gal num2_prior:', num2_prior)

    priors = [{
        'num': num1_prior,
        'mean': mean1_prior,
        'sigma': sigma1_prior,
    }]
    if ngauss == 2:
        priors.append({
            'num': num2_prior,
            'mean': mean2_prior,
            'sigma': sigma2_prior,
        })

    return priors


def _star_mean1_vs_rmag(*, rmag):
    from .fitting import exp_func
    return exp_func(
        [-1.7368e-08, 10.9409, 1.24299],
        rmag,
    )


def _star_mean2_vs_rmag(*, rmag):
    from .fitting import exp_func
    return exp_func(
        [5.52905e-07, 14.0743, 1.12144],
        rmag,
    )


def get_star_priors(*, rng, data, rmag, nstar, nstar_err):
    """
    get star priors.  The gaussian pdfs are used for generating
    guesses with bounds, only the bounds themselves are used in
    the EM algorithm

    """
    frac1 = 0.84
    frac2 = 1 - frac1


    # number in each gaussian
    num1 = frac1*nstar
    num1_err = frac1*nstar_err
    num1_prior = GaussianPrior(
        mean=num1,
        sigma=num1_err,
        bounds=[0.95*num1, 1.05*num1],
        # bounds=[0.1*num1, 2*num1],
        rng=rng,
    )
    num2 = frac2*nstar
    num2_err = frac2*nstar_err
    num2_prior = GaussianPrior(
        mean=num2,
        sigma=num2_err,
        bounds=[0.95*num2, 1.05*num2],
        # bounds=[0.1*num2, 2*num2],
        rng=rng,
    )

    # means
    # mean1 = np.interp(rmag, data['rmag_centers'], data['star_means'][:, 0])
    mean_sigma = 0.0001
    mean1 = _star_mean1_vs_rmag(rmag=rmag)
    mean2 = _star_mean2_vs_rmag(rmag=rmag)
    mean1_prior = GaussianPrior(
        mean=mean1,
        # sigma=abs(mean_sigma_frac*mean1),
        sigma=mean_sigma,
        # bounds=[mean1 - 0.5*abs(mean1), mean1 + 0.5*abs(mean1)],
        bounds=[mean1 - mean_sigma, mean1 + mean_sigma],
        rng=rng,
    )

    # mean2 = np.interp(rmag, data['rmag_centers'], data['star_means'][:, 1])
    mean2_prior = GaussianPrior(
        mean=mean2,
        sigma=mean_sigma,
        # bounds=[mean2 - 0.5*abs(mean2), mean2 + 0.1*abs(mean2)],
        bounds=[mean2 - mean_sigma, mean2 + mean_sigma],
        rng=rng,
    )

    # need to leave this fairly free to allow for seeing differences
    sigma1 = np.interp(rmag, data['rmag_centers'], data['star_sigmas'][:, 0])
    sigma_sigma_frac = 0.3
    sigma1_prior = GaussianPrior(
        mean=sigma1,
        sigma=abs(sigma_sigma_frac*sigma1),
        bounds=[0.7*sigma1, 1.3*sigma1],
        rng=rng,
    )

    sigma2 = np.interp(rmag, data['rmag_centers'], data['star_sigmas'][:, 1])

    sigma2_prior = GaussianPrior(
        mean=sigma2,
        sigma=abs(sigma_sigma_frac*sigma2),
        bounds=[0.7*sigma2, 1.3*sigma2],
        rng=rng,
    )

    print()
    print('star mean1_prior:', mean1_prior)
    print('star sigma1_prior:', sigma1_prior)
    print('star num1_prior:', num1_prior)

    print('star mean2_prior:', mean2_prior)
    print('star sigma2_prior:', sigma2_prior)
    print('star num2_prior:', num2_prior)

    priors = [
        {'num': num1_prior,
         'mean': mean1_prior,
         'sigma': sigma1_prior},
        {'num': num2_prior,
         'mean': mean2_prior,
         'sigma': sigma2_prior},
    ]

    return priors


class Fitter(object):
    def __init__(
        self,
        *,
        data,
        star_priors,
        gal_priors,
        rng,
    ):

        self.rng = rng
        self.data = np.array(data, copy=False, dtype='f8')
        self.maxiter = 2000
        self.tol = 1.0e-6

        self.star_priors = star_priors
        self.gal_priors = gal_priors
        self.priors = self.star_priors + self.gal_priors

        self.star_ngauss = len(star_priors)
        self.gal_ngauss = len(gal_priors)
        self.ngauss = self.star_ngauss + self.gal_ngauss

        self.num_low = np.array([
            p['num'].bounds[0] for p in self.priors
        ])
        self.num_high = np.array([
            p['num'].bounds[1] for p in self.priors
        ])

        self.mean_low = np.array([
            p['mean'].bounds[0] for p in self.priors
        ])
        self.mean_high = np.array([
            p['mean'].bounds[1] for p in self.priors
        ])

        self.sigma_low = np.array([
            p['sigma'].bounds[0] for p in self.priors
        ])
        self.sigma_high = np.array([
            p['sigma'].bounds[1] for p in self.priors
        ])

        # will get repeated for all
        # self.sigma_low = 1.0e-5

    def go(self, ntry=100):

        for i in range(ntry):
            self.guess = self.make_guess()

            print('guess')
            cem.gmix_print(self.guess)

            self.gmix, self.result = cem.run_em(
                data=self.data,
                guess=self.guess,
                maxiter=self.maxiter,
                tol=self.tol,
                num_low=self.num_low,
                num_high=self.num_high,
                mean_low=self.mean_low,
                mean_high=self.mean_high,
                sigma_low=self.sigma_low,
                sigma_high=self.sigma_high,
            )

            if self.result['converged']:
                break

        # cem.gmix_print(self.gmix)
        # self.gmix['num'] *= self.data.size
        # cem.gmix_print(self.gmix)
        if self.result['converged']:
            self.result['flags'] = 0
        else:
            self.result['flags'] = 1

    def make_guess(self):
        gmix = cem.make_gmix(self.ngauss)
        for i in range(self.ngauss):
            prior = self.priors[i]
            num = prior['num'].sample()
            mean = prior['mean'].sample()
            sigma = prior['sigma'].sample()

            cem.gauss_set(
                gmix[i],
                num,
                mean,
                sigma,
            )
        # nsum = gmix['num'].sum()
        # for i in range(self.ngauss):
        #     gauss = gmix[i]
        #     cem.gauss_set(
        #         gauss,
        #         gauss['num']/nsum,
        #         gauss['mean'],
        #         gauss['sigma'],
        #     )
        #
        return gmix

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


def replace_ext(fname, old_ext, new_ext):
    new_fname = fname.replace(old_ext, new_ext)
    assert new_fname != fname
    return new_fname


def plot_vs_rmag(data, type, dofits=False, show=False):
    """
    make a plot of gaussian mixture parameters vs the central
    magnitude of the bin
    """
    import hickory
    if type == 'star':
        start = 0
        end = 2
        label = 'stars'
    else:
        label = 'galaxies'
        start = 2
        end = 4

    tab = hickory.Table(
        nrows=2,
        ncols=2,
    )

    # tab[1, 1].axis('off')

    # tab[1, 1].ntext(
    #     0.5, 0.5,
    #     label,
    #     verticalalignment='center',
    #     horizontalalignment='center',
    #     fontsize=16,
    # )

    xlim = (15, 25)
    tab.suptitle(label, fontsize=15)
    tab[0, 0].set(
        xlabel='r mag',
        ylabel='weight',
        xlim=xlim
    )
    tab[0, 1].set(
        xlabel='r mag',
        ylabel='relweight',
        xlim=xlim,
    )
    tab[1, 0].set(
        xlabel='r mag',
        ylabel='mean',
        xlim=xlim,
    )

    tab[1, 1].set(
        xlabel='r mag',
        ylabel=r'$\sigma$',
        xlim=xlim,
    )

    centers = data['rmag']
    gmixes = data['gmix']

    wsums = gmixes['weight'][:, start:end].sum(axis=1)

    for igauss in range(start, end):
        weights = gmixes['weight'][:, igauss]
        relweights = weights/wsums

        means = gmixes['mean'][:, igauss]
        sigmas = gmixes['sigma'][:, igauss]

        tab[0, 0].curve(centers, weights, marker='o', markersize=2)
        tab[0, 1].curve(centers, relweights, marker='o', markersize=2)

        if dofits and type == 'gal':
            from .fitting import fit_erf, erf_func

            if igauss == 2:
                ftype = 'falling'
                guess = [0.6, 0.95, 20, 1]
            else:
                ftype = 'rising'
                guess = [0.05, 0.4, 20, 1]

            res = fit_erf(centers, relweights, guess, ftype)
            print(igauss, 'erf pars:', res['pars'])

            tab[0, 1].curve(
                centers, erf_func(res['pars'], centers, ftype),
                color='red',
            )

        # if type == 'gal':
        #     poly = np.poly1d(np.polyfit(centers, means, 2))
        #     print(poly)
        #     tab[0, 1].curve(centers, poly(centers), color='black')

        tab[1, 0].curve(centers, means, marker='o', markersize=1.5)
        tab[1, 1].curve(centers, sigmas, marker='o', markersize=1.5)

    if show:
        tab.show()

    return tab
    #
    # if output is not None:
    #     print('writing:', output)
    #     tab.savefig(output, dpi=100)
    #


def make_output(num):
    dt = [
        ('rmag', 'f8'),
        ('rmagmin', 'f8'),
        ('rmagmax', 'f8'),
        ('gmix', cem.GAUSS_DTYPE, 4),
    ]

    return np.zeros(num, dtype=dt)


def fit_conc_pdf(
    *,
    data, prior_file, rmag_index, seed, output,
    training,
    show=False,
):

    rng = np.random.RandomState(seed)

    print('will write to:', output)
    pdf_file = replace_ext(output, '.npy', '-plots.pdf')
    print('writing to pdf file:', pdf_file)
    pdf = PdfPages(pdf_file)

    data = data[data['flags'] == 0]

    prior_data = fitsio.read(prior_file)

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

    outdata = make_output(len(edges))

    rmag_centers = np.array(
        [0.5*(rmagmin + rmagmax) for rmagmin, rmagmax in edges]
    )

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

    # gal_num_pars = galamp.fit_exp(
    #     rmag=data['psf_mag'][:, rmag_index],
    #     conc=data['conc'],
    #     show=show,
    # )
    #
    # init_ngal = galamp.exp_func(gal_num_pars, rmag_centers)
    # init_ngal_err = np.sqrt(init_ngal)

    # init_ngal_err *= 10
    # init_nstar_err *= 10

    ngal_list = np.zeros(len(edges))
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
        star_priors = get_star_priors(
            rng=rng, data=prior_data, rmag=rmag,
            nstar=nstar_predicted, nstar_err=init_nstar_err[i],
        )

        # if rmag < 18.0:
        #     gal_ngauss = 1
        # else:
        #     gal_ngauss = 2

        gal_ngauss = 2

        ngal_predicted = w.size - nstar_predicted
        if ngal_predicted < 3:
            ngal_predicted = 3

        gal_priors = get_gal_priors(
            rng=rng, data=prior_data, rmag=rmag,
            ngal=ngal_predicted,
            ngal_err=np.sqrt(ngal_predicted),
            ngauss=gal_ngauss,
            training=training,
        )

        fitter = Fitter(
            data=data['conc'][w],
            star_priors=star_priors,
            gal_priors=gal_priors,
            rng=rng,
        )

        fitter.go()

        res = fitter.result
        print('final')
        cem.gmix_print(fitter.gmix)
        print(res)
        assert res['flags'] == 0, 'failed to converge'

        nstar_meas = fitter.gmix['num'][:2].sum()
        ngal_meas = fitter.gmix['num'][2:].sum()
        nobj_meas = fitter.gmix['num'].sum()

        ngal_list[i] = ngal_meas
        print('true num: %g meas num: %g' % (w.size, nobj_meas))
        print('nstar pred: %g nstar meas: %g' % (nstar_predicted, nstar_meas))
        print('ngal pred: %g ngal meas: %g' % (ngal_predicted, ngal_meas))

        if rmag > 23.5:
            plt = fitter.plot(title=label, legend=True, show=show)
        else:
            plt = fitter.plot3(label=label, show=show)

        pdf.savefig(figure=plt)

        outdata['rmagmin'][i] = rmagmin
        outdata['rmagmax'][i] = rmagmax
        outdata['rmag'][i] = rmag
        outdata['gmix'][i] = fitter.gmix

    plt = plot_vs_rmag(outdata, 'star', show=show)
    pdf.savefig(plt)
    plt = plot_vs_rmag(outdata, 'gal', show=show)
    pdf.savefig(plt)

    print('closing pdf file:', pdf_file)
    pdf.close()
    print('writing:', output)
    np.save(output, outdata)
