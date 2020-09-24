import numpy as np
import ngmix
import fitsio
import esutil as eu
from esutil.numpy_util import between
from . import staramp

from .constants import MAGZP


def inject_star_into_obs(*, rng, obs, star_flux, poisson=True):
    """
    inject a star into the observation


    Parameters
    ----------
    rng: np.RandomState
        The random number generator
    obs: ngmix.Observation
        Observation to be replaced with the injected star.
    star_flux: float
        Flux for the star
    poisson: bool
        If True, the signal is a poisson deviate.  Default True

    Returns
    -------
    new_obs: ngmix.Observation
        The new observation
    """
    new_image = obs.psf.image.copy()
    weight = obs.weight.copy()

    # new_image *= obs.meta['psf_flux']/new_image.sum()
    new_image *= star_flux/new_image.sum()

    if poisson:
        # now in electrons, which are poisson distributed
        simage = new_image / obs.meta['scale']
        simage.clip(min=0, max=None, out=simage)

        new_image = rng.poisson(lam=simage).astype('f8')

    weight_for_noise = weight.copy()
    w = np.where(weight_for_noise <= 0)
    if w[0].size > 0:
        weight_for_noise[w] = weight.max()

    noise = np.sqrt(1.0/weight_for_noise)
    new_image += noise*rng.normal(scale=1, size=new_image.shape)

    new_obs = ngmix.Observation(
        new_image,
        weight=weight,
        jacobian=obs.psf.jacobian,
        psf=obs.psf,
    )

    return new_obs


def inject_star_into_obslist(*, rng, obslist, star_flux):
    """
    inject a star into the observations


    Parameters
    ----------
    rng: np.RandomState
        The random number generator
    obs: ngmix.ObsList
        Observations to be replaced with the injected star.

    Returns
    -------
    new_obs_list: ngmix.ObsList
        The new observations
    """

    new_obslist = ngmix.ObsList()
    for obs in obslist:
        new_obs = inject_star_into_obs(rng=rng, obs=obs, star_flux=star_flux)
        new_obslist.append(new_obs)

    return new_obslist


class Stars(object):
    """
    create stars for injection

    r band magnitudes are drawn from a power law distribution.  For the
    power law parameters see the staramp module

    Parameters
    ----------
    fname: str
        Path to the stars file
    rng: np.random.RandomState
        random number generator
    max_inject_rmag: float
        maximum magintude for injections.  The minimum mag is the "magoff"
        magoffset for the power law PDF of r-band magnitudes
    """
    def __init__(self, *, fname, rng, max_inject_rmag):
        self.rng = rng
        data = fitsio.read(fname)

        gmr = data['psf_mag'][:, 0] - data['psf_mag'][:, 1]
        rmi = data['psf_mag'][:, 1] - data['psf_mag'][:, 2]
        imz = data['psf_mag'][:, 2] - data['psf_mag'][:, 3]
        w, = np.where(
            between(gmr, -1, 3) &
            between(rmi, -1, 3) &
            between(imz, -1, 2)
        )

        self.data = data[w]
        self.gmr = gmr[w]
        self.rmi = rmi[w]
        self.imz = imz[w]

        self.bright, = np.where(self.data['psf_mag'][:, 1] < 21)

        mrng = [staramp.MAGOFF, max_inject_rmag]
        self.generator = eu.random.Generator(
            lambda x: (x - staramp.MAGOFF)**staramp.SLOPE,
            xrange=mrng,
            nx=100,
        )

    def sample(self, num):
        """
        sample a mag from the full set of data, but color only from
        the bright subset
        """

        samples = np.zeros(num, dtype=self.data.dtype)

        rmag = self.generator.sample(num)

        for i in range(num):
            samples[i] = self.sample_one(rmag[i])

        return samples

    def sample_one(self, rmag):
        """
        sample a star with the given r-band magnitude and a
        random color from the bright end of the star catalog

        Parameters
        ----------
        rmag: float
            r-band magnitude
        """

        # get color from the bright end
        ind = self.rng.randint(0, self.bright.size)
        ind = self.bright[ind]
        sample = self.data[ind].copy()

        gmr = self.gmr[ind]
        rmi = self.rmi[ind]
        imz = self.imz[ind]

        gmag = gmr + rmag
        imag = -rmi + rmag
        zmag = -imz + imag

        sample['psf_mag'][0] = gmag
        sample['psf_mag'][1] = rmag
        sample['psf_mag'][2] = imag
        sample['psf_mag'][3] = zmag

        sample['psf_flux'][0] = 10**((MAGZP - gmag)/2.5)
        sample['psf_flux'][1] = 10**((MAGZP - rmag)/2.5)
        sample['psf_flux'][2] = 10**((MAGZP - imag)/2.5)
        sample['psf_flux'][3] = 10**((MAGZP - zmag)/2.5)

        return sample
