import numpy as np
import esutil as eu
from esutil.numpy_util import between

MAGOFF = 13.5
SLOPE = 1.5


def predict(*, rmag, amp):
    return amp * (rmag - MAGOFF)**SLOPE


def get_amp(*, rmag, conc, show=False, output=None):
    """
    get the amplitude of the stellar locus assuming a
    power law distribution A (mag - 13.5)**1.5

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
    amp: float
        Amplitude
    """
    # magmin, magmax = 16.0, 19.5
    # nbin = 7
    magmin, magmax = 16.5, 19.5
    nbin = 6

    cmin, cmax = -0.0005, 0.0003
    w, = np.where(
        between(rmag, magmin, magmax)
        &
        between(conc, cmin, cmax)
    )

    hd = eu.stat.histogram(
        rmag[w],
        min=magmin, max=magmax, nbin=nbin,
        more=True,
    )
    herr = np.sqrt(hd['hist'])

    pv = predict(rmag=hd['center'], amp=1.0)
    amp = (hd['hist']/pv).mean()
    print('amp:', amp)

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
            xlim=(16, 20),
            ylim=(-0.002, 0.005)
        )

        plt[0].set(
            xlabel='psf mag r',
            ylabel='conc',
        )

        predicted = amp*pv
        plt[1].errorbar(
            hd['center'],
            hd['hist'],
            yerr=herr,
        )
        plt[1].curve(hd['center'], predicted)
        xvals = np.linspace(13.9, 21)
        plt[1].curve(
            xvals,
            predict(rmag=xvals, amp=amp),
        )
        plt[1].set(
            xlabel='psf mag r',
            ylabel='Number',
            xlim=(13.5, 21),
        )

        if show:
            plt.show()

        if output is not None:
            output = output.replace('.png', '-fit.png')
            print('writing:', output)
            plt.savefig(output)

    return amp
