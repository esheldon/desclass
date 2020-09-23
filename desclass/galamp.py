import numpy as np
import esutil as eu
from esutil.numpy_util import between
from ngmix.fitting import run_leastsq, print_pars

MAGOFF = 13.5
SLOPE = 1.5
# MAGMIN, MAGMAX = 16.5, 19.5
# NBIN = 6
# MAGMIN, MAGMAX = 16.5, 21.5
# NBIN = 10
MAGMIN, MAGMAX = 16.5, 20.5
NBIN = 8
CMIN, CMAX = 0.005, 0.03


def predict(*, rmag, amp, amp_err=None):
    """
    predict the rmag hist for the predefined mag bins

        num = amp * (rmag - MAGOFF)**SLOPE

    Parameters
    ----------
    rmag: float or array
        r band magnitudes in bins
    amp: float
        amplitude of the relation

    Returns
    -------
    the predicted number in each rmag bin
    """
    num = amp * (rmag - MAGOFF)**SLOPE
    if amp_err is not None:
        num_err = amp_err * (rmag - MAGOFF)**SLOPE
        return num, num_err
    else:
        return num


def calculate_amp(*, rmag, num):
    """
    calculate the amplitude assuming a relationship of
        num = amp * (rmag - MAGOFF)**SLOPE


    Parameters
    ----------
    rmag: array
        rmag in bins
    num: array
        number in each bin

    Returns
    -------
    amp, amp_err: float, float
        Amplitude and uncertainty
    """
    pv = predict(rmag=rmag, amp=1.0)
    amp = (num/pv).mean()
    amp_err = (num/pv).std()
    return amp, amp_err


def exp_func(pars, x):
    amp = pars[0]
    off = pars[1]
    sigma = pars[2]
    platform = pars[3]

    try:
        _ = len(x)
        scalar = False
    except TypeError:
        scalar = True
        x = np.array(x, ndmin=1)

    model = np.zeros(x.size) + platform
    w, = np.where(x > off)
    if w.size > 0:
        arg = ((x[w] - off)/sigma)**2
        model[w] += amp * (np.exp(arg) - 1)

    if scalar:
        model = model[0]

    return model


def fit_exp_binned(x, y, yerr, guess):
    # assume quadratic

    def loss(pars):
        model = exp_func(pars, x)
        return (model - y)/yerr

    return run_leastsq(
        loss,
        np.array(guess),
        0,
    )


def fit_exp(*, rmag, conc, show=False, output=None):
    """
    get the amplitude of the stellar locus assuming a
    power law distribution A (mag - 13.5)**1.5  The data are binned
    and clipped, and sent to the calculate_amp function.

    Parameters
    ----------
    rmag: array
        psf magnitude, should be r band
    conc: array
        concentration parameters for stars
    show: bool
        If True, show a plot
    output: string, optional
        If sent, write a plot file with that name

    Returns
    -------
    amp, amp_err: float, float
        Amplitude and uncertainty
    """

    w, = np.where(
        between(rmag, MAGMIN, MAGMAX)
        &
        between(conc, CMIN, CMAX)
    )

    hd = eu.stat.histogram(
        rmag[w],
        min=MAGMIN, max=MAGMAX, nbin=NBIN,
        more=True,
    )

    num = hd['hist']
    num_err = np.sqrt(num)
    guess = [1000, 18, 4, num[:4].mean()]
    res = fit_exp_binned(hd['center'], num, num_err, guess)
    print_pars(res['pars'], front='pars: ')
    print_pars(res['pars_err'], front='pars: ')

    if res['flags'] != 0:
        raise RuntimeError('fit failed')

    if show or output is not None:
        import hickory
        # alpha = 0.5
        plt = hickory.Table(2, 1, figsize=(8, 7))

        plt[0].plot(
            rmag,
            conc,
            marker='.',
            alpha=0.5,
        )
        plt[0].plot(
            rmag[w],
            conc[w],
            marker='.',
            markeredgecolor='black',
            alpha=0.5,
        )

        plt[0].set(
            xlim=(MAGMIN-0.5, MAGMAX+0.5),
            ylim=(-0.0005, CMAX),
        )

        plt[0].set(
            xlabel='psf mag r',
            ylabel='conc',
        )

        plt[1].errorbar(
            hd['center'],
            num,
            yerr=num_err,
        )

        predicted = exp_func(res['pars'], hd['center'])

        eu.misc.colprint(num, predicted, num_err,
                         format='%20s',
                         names=['num', 'pred', 'num_err'])

        plt[1].curve(hd['center'], predicted)
        xvals = np.linspace(13.9, MAGMAX+0.5)
        plt[1].curve(
            xvals,
            exp_func(res['pars'], xvals),
        )
        plt[1].set(
            xlabel='psf mag r',
            ylabel='Number',
            xlim=(MAGMIN-0.5, MAGMAX+0.5),
            yscale='log',
            # xlim=(13.5, MAGMAX+0.5),
        )

        if show:
            plt.show()

        if output is not None:
            print('writing:', output)
            plt.savefig(output)

    return res['pars']
