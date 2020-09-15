import numpy as np
import esutil as eu
from esutil.numpy_util import between
import fitsio
import hickory
# import scipy.optimize
from ngmix.fitting import run_leastsq
from ngmix import print_pars
from . import staramp
from . import galamp


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


def _gal_mean1_vs_rmag(*, rmag):
    return -4.729e-05 * rmag**2 + 0.001028 * rmag + 0.015


def _gal_mean2_vs_rmag(*, rmag):
    return -4.729e-05 * rmag**2 + 0.001028 * rmag + 0.006156


def get_gal_priors(*, rng, data, rmag, ngal, ngal_err, ngauss):
    if ngauss == 1:
        frac1 = 1
    elif ngauss == 2:
        # frac1 = 0.6
        frac1 = 0.5
        frac2 = 1 - frac1
    else:
        raise ValueError('ngauss should be 1 or 2')

    # num_sigma_frac = 0.1
    if ngauss == 1:
        mean_sigma_frac = 1
    else:
        mean_sigma_frac = 0.3

    covar_sigma_frac = 0.3
    # num_sigma_frac = 1
    # mean_sigma_frac = 1
    # covar_sigma_frac = 1

    # number in each gaussian
    num1 = frac1*ngal
    # num1_err = frac1*ngal_err
    num1_err = num1
    num1_prior = GaussianPrior(
        mean=num1,
        sigma=num1_err,
        # sigma=abs(num_sigma_frac*num1),
        # sigma=np.sqrt(num1),
        bounds=[0.1*num1, 2*num1],
        rng=rng,
    )
    if ngauss == 2:
        num2 = frac2*ngal
        # num2_err = frac2*ngal_err
        num2_err = num2
        num2_prior = GaussianPrior(
            mean=num2,
            sigma=num2_err,
            # sigma=abs(num_sigma_frac*num2),
            # sigma=np.sqrt(num2),
            bounds=[0.1*num2, 2*num2],
            rng=rng,
        )

    # mean of gaussians
    if ngauss == 1:
        mean1 = 0.01
    else:
        mean1 = _gal_mean1_vs_rmag(rmag=rmag)

    mean1_prior = GaussianPrior(
        mean=mean1,
        sigma=abs(mean_sigma_frac*mean1),
        bounds=[0.005, 0.02],
        rng=rng,
    )

    if ngauss == 2:
        mean2 = _gal_mean2_vs_rmag(rmag=rmag)
        mean2_prior = GaussianPrior(
            mean=mean2,
            sigma=abs(mean_sigma_frac*mean2),
            bounds=[0.005, 0.02],
            rng=rng,
        )

    # no extrapolation for covars, just to linear interpolation
    # and use endpoints for those outside

    covar1 = np.interp(rmag, data['rmag_centers'], data['gal_covars'][:, 0])
    covar1_prior = GaussianPrior(
        mean=covar1,
        sigma=abs(covar_sigma_frac*covar1),
        bounds=[0.1*covar1, 2*covar1],
        rng=rng,
    )

    if ngauss == 2:
        covar2 = np.interp(
            rmag, data['rmag_centers'], data['gal_covars'][:, 1],
        )
        covar2_prior = GaussianPrior(
            mean=covar2,
            sigma=abs(covar_sigma_frac*covar2),
            bounds=[0.1*covar2, 2*covar2],
            rng=rng,
        )

    print()
    print('gal mean1_prior:', mean1_prior)
    print('gal covar1_prior:', covar1_prior)
    print('gal num1_prior:', num1_prior)

    if ngauss == 2:
        print('gal mean2_prior:', mean2_prior)
        print('gal covar2_prior:', covar2_prior)
        print('gal num2_prior:', num2_prior)

    priors = [
        mean1_prior,
        covar1_prior,
        num1_prior,
    ]
    if ngauss == 2:
        priors += [
            mean2_prior,
            covar2_prior,
            num2_prior,
        ]

    return priors


def get_star_priors(*, rng, data, rmag, nstar, nstar_err):
    frac1 = 0.85
    frac2 = 1 - frac1

    # mean_sigma_frac = 0.1
    # mean_sigma_frac = 0.3
    # covar_sigma_frac = 0.3
    mean_sigma_frac = 0.5
    covar_sigma_frac = 0.5

    # number in each gaussian
    num1 = frac1*nstar
    num1_err = frac1*nstar_err
    num1_prior = GaussianPrior(
        mean=num1,
        # sigma=abs(num_sigma_frac*num1),
        sigma=num1_err,
        bounds=[0.1*nstar, 2*nstar],
        rng=rng,
    )
    num2 = frac2*nstar
    num2_err = frac2*nstar_err
    num2_prior = GaussianPrior(
        mean=num2,
        sigma=num2_err,
        # sigma=abs(num_sigma_frac*num2),
        # sigma=np.sqrt(num2),
        bounds=[0.1*nstar, 2*nstar],
        rng=rng,
    )

    # means
    mean1 = np.interp(rmag, data['rmag_centers'], data['star_means'][:, 0])
    mean1_prior = GaussianPrior(
        mean=mean1,
        sigma=abs(mean_sigma_frac*mean1),
        bounds=[-0.002, 0.001],
        rng=rng,
    )

    mean2 = np.interp(rmag, data['rmag_centers'], data['star_means'][:, 1])
    mean2_prior = GaussianPrior(
        mean=mean2,
        sigma=abs(mean_sigma_frac*mean2),
        bounds=[-0.002, 0.001],
        rng=rng,
    )

    # covariances
    covar1 = np.interp(rmag, data['rmag_centers'], data['star_covars'][:, 0])
    covar1_prior = GaussianPrior(
        mean=covar1,
        sigma=abs(covar_sigma_frac*covar1),
        bounds=[0.1*covar1, 2*covar1],
        rng=rng,
    )

    covar2 = np.interp(rmag, data['rmag_centers'], data['star_covars'][:, 1])
    covar2_prior = GaussianPrior(
        mean=covar2,
        sigma=abs(covar_sigma_frac*covar2),
        bounds=[0.1*covar2, 2*covar2],
        rng=rng,
    )

    print()
    print('star mean1_prior:', mean1_prior)
    print('star covar1_prior:', covar1_prior)
    print('star num1_prior:', num1_prior)

    print('star mean2_prior:', mean2_prior)
    print('star covar2_prior:', covar2_prior)
    print('star num2_prior:', num2_prior)

    return [
        mean1_prior,
        covar1_prior,
        num1_prior,
        mean2_prior,
        covar2_prior,
        num2_prior,
    ]


class Fitter(object):
    def __init__(
        self,
        *,
        x, y, yerr,
        star_priors,
        gal_priors,
        rng,
    ):

        self.rng = rng

        self.x = x
        self.y = y
        self.yerr = yerr.clip(min=1)

        self.binsize = x[1] - x[0]

        assert len(star_priors) % 3 == 0
        assert len(gal_priors) % 3 == 0

        self.star_priors = star_priors
        self.gal_priors = gal_priors
        self.priors = self.star_priors + self.gal_priors

        self.star_ngauss = len(star_priors)//3
        self.gal_ngauss = len(gal_priors)//3
        self.ngauss = self.star_ngauss + self.gal_ngauss

        self.n_prior_pars = 3*self.ngauss
        assert len(self.priors) == self.n_prior_pars

        self.fdiff_size = self.n_prior_pars + self.x.size

        # self.bounds = [
        #     (None, None),
        #     # (1.0e-9, None),
        #     (1.0e-10, None),
        #     (0, None),
        # ]*self.ngauss

        self.bounds = []
        for prior in self.priors:
            self.bounds += [prior.bounds]

    def go(self, ntry=100):

        ngauss = self.ngauss

        # ysum = self.y.sum()
        npars_per = 3

        for i in range(ntry):
            self.guess = np.zeros(npars_per*ngauss)

            for ip, prior in enumerate(self.priors):
                self.guess[ip] = prior.sample(sigma_factor=1)
                # self.guess[ip] = prior.sample(sigma_factor=0.1)

            print_pars(self.guess, front='guess: ')
            self.result = run_leastsq(
                self._errfunc,
                self.guess,
                self.n_prior_pars,
                bounds=self.bounds,
                maxfev=4000,
            )
            if self.result['flags'] == 0:
                break

    def _scale_leastsq_cov(self, pars, pcov):
        """
        Scale the covariance matrix returned from leastsq; this will
        recover the covariance of the parameters in the right units.
        """
        dof = (self.x.size-len(pars))
        s_sq = (self._errfunc(pars)**2).sum()/dof
        return pcov * s_sq

    def eval(self, pars, x=None, components=None):

        if x is None:
            x = self.x

        model = np.zeros(x.size)
        npars_per = 3

        if components is None:
            components = range(self.ngauss)

        for i in components:
            start = i*npars_per
            end = (i+1)*npars_per

            tpars = pars[start:end]
            tmodel = self.eval_one(tpars, x=x)
            model[:] += tmodel
        return model

    def eval_one(self, pars, x=None):
        if x is None:
            x = self.x

        mn = pars[0]
        sigma2 = pars[1]
        amp = pars[2] * self.binsize

        arg = -0.5 * (mn - x)**2/sigma2

        return amp * np.exp(arg)/np.sqrt(2*np.pi*sigma2)

    def eval_priors(self, pars):
        """
        pars are [
            cen1, sigma^2_1, amp1,
            cen2, sigma^2_2, amp2,
            ...
        ]
        """

        assert pars.size == len(self.priors)

        priors_fdiff = np.zeros(self.n_prior_pars)

        for i, prior in enumerate(self.priors):
            priors_fdiff[i] = prior.get_fdiff(pars[i])

        return priors_fdiff

    def _errfunc(self, pars):
        # print_pars(pars, front='pars: ')

        # diff = np.zeros(self.y.size)
        model = self.eval(pars)

        diff = (model-self.y)/self.yerr

        fdiff = np.zeros(self.fdiff_size)

        fdiff[0:self.n_prior_pars] = self.eval_priors(pars)
        fdiff[self.n_prior_pars:] = diff
        return fdiff

    def plot(
        self, *,
        x=None,
        y=None,
        components=None,
        min=None,
        max=None,
        npts=None,
        data=None,
        nbin=None,
        binsize=None,
        file=None,
        dpi=100,
        show=False,
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
        npts: int, optional
            Number of points to use for the plot.  If data are sent you
            can leave this off and a suitable value will be chosen based
            on the data binsize
        data: array, optional
            Optional data to plot as a histogram
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

        pars = self.result['pars']

        plt = hickory.Plot(**plot_kws)

        dx_orig = self.x[1] - self.x[0]

        if x is not None and y is not None:
            dx_data = x[1] - x[0]
            fac = dx_data/dx_orig
        elif x is not None or y is not None:
            raise ValueError('send both x and y')
        else:
            x = self.x
            y = self.y
            dx_data = dx_orig
            fac = 1

        plt.bar(
            x,
            y,
            label='data',
            width=dx_data,
            alpha=0.5,
            color='#a6a6a6',
        )

        model = fac*self.eval(
            pars, components=components,
            x=x,
        )

        plt.curve(x, model, label='model')

        if components is None:
            components = range(self.ngauss)

        npars_per = 3
        for i in components:
            start = i*npars_per
            end = (i+1)*npars_per

            tpars = pars[start:end]
            tmodel = fac*self.eval_one(tpars, x=x)

            label = 'component %d' % i
            plt.curve(x, tmodel, label=label)

        if show:
            plt.show()

        if file is not None:
            plt.savefig(file, dpi=dpi)

        return plt


def fit_conc_hists(*, data, prior_file, rmag_index, seed):

    rng = np.random.RandomState(seed)

    data = data[data['flags'] == 0]

    prior_data = fitsio.read(prior_file)

    edges = [
        (16.5, 17.0),
        (17.0, 17.5),
        (17.5, 18.0),
        (18.0, 18.5),

        # (16, 18.5),
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
    # edges = [
    #     (23, 23.5),
    #     (23.5, 24.0),
    #     (24, 24.5),
    # ]
    #

    rmag_centers = np.array(
        [0.5*(rmagmin + rmagmax) for rmagmin, rmagmax in edges]
    )

    # initial estimate of N(mag) for stars
    initial_amp, initial_amp_err = staramp.get_amp(
        rmag=data['psf_mag'][:, rmag_index],
        conc=data['conc'],
        show=True,
    )
    gal_num_pars = galamp.fit_exp(
        rmag=data['psf_mag'][:, rmag_index],
        conc=data['conc'],
        show=True,
    )

    init_nstar, init_nstar_err = staramp.predict(
        rmag=rmag_centers,
        amp=initial_amp,
        amp_err=initial_amp_err,
    )
    init_ngal = galamp.exp_func(gal_num_pars, rmag_centers)
    # init_ngal[0] = galamp.exp_func(gal_num_pars, 18.5)

    init_ngal_err = np.sqrt(init_ngal)

    # init_ngal_err *= 10
    # init_nstar_err *= 10

    ngal_list = np.zeros(len(edges))
    for i in range(len(edges)):
        rmagmin, rmagmax = edges[i]
        rmag = 0.5*(rmagmin + rmagmax)
        label = 'rmag: [%.2f, %.2f]' % (rmagmin, rmagmax)

        binsize = 0.00005
        binsize_coarse = 0.0004
        binsize_coarser = 0.0004 * 2

        # off = 0.007
        # power = 0.5

        minconc, maxconc = -0.01, 0.025
        # minconc, maxconc = -0.0005, 0.01
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

        if rmag < 18.0:
            gal_ngauss = 1
        else:
            gal_ngauss = 2

            # print_pars(ngal_list[0:i], front='ngal_list: ')
            # gal_num_res = galamp.fit_exp_binned(
            #     rmag_centers[0:i],
            #     ngal_list[0:i],
            #     np.sqrt(ngal_list[0:i]),
            #     gal_num_pars,
            # )
            # print_pars(gal_num_pars, front='orig: ')
            # gal_num_pars = gal_num_res['pars']
            # print_pars(gal_num_pars, front='new: ')
            # init_ngal = galamp.exp_func(gal_num_pars, rmag_centers)
            # # init_ngal[0] = galamp.exp_func(gal_num_pars, 18.5)
            #
            # init_ngal_err = np.sqrt(init_ngal)

        # gal_ngauss = 2
        ngal_predicted = w.size - nstar_predicted
        if ngal_predicted < 3:
            ngal_predicted = init_ngal[i]
            if ngal_predicted < 3:
                ngal_predicted = 3

        gal_priors = get_gal_priors(
            rng=rng, data=prior_data, rmag=rmag,
            ngal=ngal_predicted, ngal_err=init_ngal_err[i],
            ngauss=gal_ngauss,
        )

        hd = eu.stat.histogram(
            data['conc'][w],
            min=minconc,
            max=maxconc,
            binsize=binsize,
            more=True,
        )
        plt = hickory.Plot(title=label, xlabel='concentration')
        plt.bar(hd['center'], hd['hist'], width=binsize)
        plt.show()
        hd_coarse = eu.stat.histogram(
            data['conc'][w],
            min=minconc,
            max=maxconc,
            binsize=binsize_coarse,
            more=True,
        )
        hd_coarser = eu.stat.histogram(
            data['conc'][w],
            min=minconc,
            max=maxconc,
            binsize=binsize_coarser,
            more=True,
        )

        x = hd['center']
        y = hd['hist']
        yerr = np.sqrt(hd['hist'])
        fitter = Fitter(
            x=x, y=y, yerr=yerr,
            star_priors=star_priors,
            gal_priors=gal_priors,
            rng=rng,
        )

        fitter.go()

        res = fitter.result
        print_pars(res['pars'], front='pars: ')
        print_pars(res['pars_err'], front='perr: ')
        pars = res['pars']

        nstar_meas = pars[2] + pars[5]
        if gal_ngauss == 1:
            nobj_meas = pars[2] + pars[5] + pars[8]
            ngal_meas = res['pars'][-1]
        else:
            nobj_meas = pars[2] + pars[5] + pars[8] + pars[11]
            ngal_meas = res['pars'][-1] + res['pars'][-4]

        ngal_list[i] = ngal_meas
        print('true num: %g meas num: %g' % (w.size, nobj_meas))
        print('nstar pred: %g nstar meas: %g' % (nstar_predicted, nstar_meas))
        print('ngal pred: %g ngal meas: %g' % (ngal_predicted, ngal_meas))

        if res['flags'] == 0:
            fitter.plot(
                figsize=(10, 7.5),
                x=hd_coarse['center'],
                y=hd_coarse['hist'],
                xlabel='concentration',
                legend=True, show=True,
                title=label,
                xlim=(-0.005, 0.025),
            )

            splt = fitter.plot(
                figsize=(10, 7.5),
                legend=True, show=False,
                title=label,
                xlim=(-0.003, 0.003),
            )
            splt.show()

            cmin, cmax = 0.003, 0.025
            w, = np.where(between(hd_coarser['center'], cmin, cmax))
            ylim = [0, 1.1*hd_coarser['hist'][w].max()]

            fitter.plot(
                figsize=(10, 7.5),
                x=hd_coarser['center'],
                y=hd_coarser['hist'],
                xlabel='concentration',
                legend=True, show=True,
                title=label,
                xlim=(0, cmax),
                ylim=ylim,
            )
