import ngmix


def do_psf_flux(*, obslist):
    """
    get psf fluxe for the input observations. Set the psf flux in
    all the Observation meta data

    Parameters
    ----------
    obs: ngmix.ObsList
        Observations to be processed
    """
    fitter = ngmix.galsimfit.GalsimTemplateFluxFitter(
        obslist,
        draw_method='no_pixel',  # already has pixel in it
    )
    fitter.go()
    res = fitter.get_result()
    for obs in obslist:
        obs.meta['psf_flags'] = res['flags']
        obs.meta['psf_flux'] = res['flux']
        obs.meta['psf_flux_err'] = res['flux_err']
