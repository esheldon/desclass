import numpy as np
import esutil as eu
from esutil.numpy_util import between
import fitsio
import hickory
# import scipy.optimize
from ngmix import print_pars
from . import staramp
from . import galamp
import scipy.stats


class StarGalaxyModel(object):
    def __init__(self, pars):
        self.pars = pars

        self.amp = pars[0]
        a = pars[1]
        loc = pars[2]
        scale = pars[3]

        self.gal1_amp = pars[4]
        gal1_mean = pars[5]
        gal1_sigma = pars[6]

        self.gal2_amp = pars[7]
        gal2_mean = pars[8]
        gal2_sigma = pars[9]

        self.sk = scipy.stats.skewnorm(a=a, loc=loc, scale=scale)
        self.g1 = scipy.stats.norm(loc=gal1_mean, scale=gal1_sigma)
        self.g2 = scipy.stats.norm(loc=gal2_mean, scale=gal2_sigma)

        self.amps = np.array([self.amp, self.gal1_amp, self.gal2_amp])
        self.weights = self.amps/self.amps.sum()
        self.pdfs = [self.sk, self.g1, self.g2]

    def pdf(self, x):

        vals = np.zeros(len(x))

        for pdf, amp in zip(self.pdfs, self.amps):
            these_vals = pdf.pdf(x)
            these_vals *= amp

            vals += these_vals

        return vals

    def sample(self, n=None, rng=None):

        if n is None:
            scalar = True
            n = 1
        else:
            scalar = False

        if rng is None:
            rng = np.random.RandomState()

        component_sizes = rng.multinomial(n, self.weights)

        samples = np.zeros(n)

        start = 0
        for i, size in enumerate(component_sizes):
            these_samples = self.pdfs[i].rvs(size, random_state=rng)
            samples[start:start+size] = these_samples
            start += size

        if scalar:
            samples = samples[0]

        return samples


class StarModel(object):
    def __init__(self, pars):
        self.pars = pars

        self.amp = pars[0]
        a = pars[1]
        loc = pars[2]
        scale = pars[3]

        self.dist = scipy.stats.skewnorm(a=a, loc=loc, scale=scale)

    def pdf(self, x):
        return self.amp * self.dist.pdf(x)

    def sample(self, n=None, rng=None):
        return self.pdf.rvs(n, random_state=rng)

    def plot(
        self, *,
        x,
        fac=1,
        file=None,
        dpi=100,
        show=False,
        plt=None,
        **plot_kws
    ):
        """
        plot the model

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

        if plt is None:
            plt = hickory.Plot(**plot_kws)

        model = fac*self.pdf(x)

        plt.curve(x, model, label='model')

        if show:
            plt.show()

        if file is not None:
            plt.savefig(file, dpi=dpi)

        return plt


class PriorBase(object):
    def sample(self):
        r = []
        for prior in self.priors:
            val = prior.sample()
            r.append(val)
        return np.array(r)


class StarPrior(PriorBase):
    def __init__(
        self,
        *,
        rng,
        amp,
        amp_sigma,
        a,
        a_sigma,
        loc,
        loc_sigma,
        scale,
        scale_sigma,
    ):

        amp_prior = GaussianPrior(
            mean=amp, sigma=amp_sigma, rng=rng,
            bounds=(0.001, np.inf),
        )
        a_prior = GaussianPrior(
            mean=a, sigma=a_sigma, rng=rng,
            bounds=(-1, 4),
        )
        loc_prior = GaussianPrior(
            mean=loc, sigma=loc_sigma, rng=rng,
            bounds=(-0.007, 0.007),
        )
        scale_prior = GaussianPrior(
            mean=scale, sigma=scale_sigma, rng=rng,
            bounds=(1.0e-4, 0.01),
        )

        self.priors = [
            amp_prior,
            a_prior,
            loc_prior,
            scale_prior,
        ]


class GalaxyPrior(PriorBase):
    def __init__(
        self,
        *,
        rng,
        gal1_amp,
        gal1_mean,
        gal1_sigma,
        gal2_amp,
        gal2_mean,
        gal2_sigma,
        width=0.3,
    ):

        gal1_amp_prior = GaussianPrior(
            mean=gal1_amp, sigma=gal1_amp*width, rng=rng,
            bounds=(0, np.inf),
        )
        gal1_mean_prior = GaussianPrior(
            mean=gal1_mean, sigma=gal1_mean*width, rng=rng,
            bounds=(0.003, 0.025),
        )
        gal1_sigma_prior = GaussianPrior(
            mean=gal1_sigma, sigma=gal1_sigma*width, rng=rng,
            bounds=(0.001, 0.01),
        )

        gal2_amp_prior = GaussianPrior(
            mean=gal2_amp, sigma=gal2_amp*width, rng=rng,
            bounds=(0, np.inf),
        )
        gal2_mean_prior = GaussianPrior(
            mean=gal2_mean, sigma=gal2_mean*width, rng=rng,
            bounds=(0.003, 0.025),
        )
        gal2_sigma_prior = GaussianPrior(
            mean=gal2_sigma, sigma=gal2_sigma*width, rng=rng,
            bounds=(0.001, 0.01),
        )

        self.priors = [
            gal1_amp_prior,
            gal1_mean_prior,
            gal1_sigma_prior,

            gal2_amp_prior,
            gal2_mean_prior,
            gal2_sigma_prior,
        ]


class StarGalaxyPrior(PriorBase):
    def __init__(self, *, rng, star_prior, gal_prior):
        self.priors = star_prior.priors + gal_prior.priors


class GaussianPrior(object):
    def __init__(self, *, mean, sigma, bounds=None, rng=None):
        self.mean = mean
        self.sigma = abs(sigma)
        self.ivar = 1/sigma**2

        if bounds is None:
            bounds = [-np.inf, np.inf]
        if rng is None:
            rng = np.random.RandomState()

        self.rng = rng
        self.bounds = bounds

    def get_logprob(self, value):
        return -0.5*(value - self.mean)**2 * self.ivar

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


class Fitter(object):
    def __init__(self, *, x, y, yerr, prior, model_class):

        self.x = x
        self.y = y
        self.yerr = yerr.clip(min=1)
        self.ivar = 1.0/self.yerr**2

        self.prior = prior
        self.bounds = [p.bounds for p in self.prior.priors]

        self.model_class = model_class

    def go(self, guess):
        from scipy.optimize import fmin_l_bfgs_b

        x, f, d = fmin_l_bfgs_b(
            self.get_nlogprob,
            guess,
            bounds=self.bounds,
            approx_grad=True,
        )

        self.result = {
            'pars': x,
            'flags': d['warnflag'],
            'nfev': d['funcalls'],
        }

    def get_nlogprob(self, pars):
        return -self.get_logprob(pars)

    def get_logprob(self, pars):
        return self.get_loglike(pars) + self.get_log_prior(pars)

    def get_loglike(self, pars):
        mod = self.get_model(pars)

        chi2 = (mod - self.y)**2 * self.ivar
        chi2 = chi2.sum()
        return -0.5 * chi2

    def get_log_prior(self, pars):

        logprob = 0.0

        for i in range(pars.size):
            prior = self.prior.priors[i]

            logprob += prior.get_logprob(pars[i])

        return logprob

    def get_model(self, pars, x=None):
        if x is None:
            x = self.x

        # e.g. StarGalaxyModel, StarModel, GalaxyModel
        mod = self.model_class(pars)
        return mod.pdf(x)

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
