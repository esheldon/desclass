# how much to dilate the weight function T for second measurement
DILATE = 1.05

#
#  DES specific things
#

# radius in pixels for trimming stamps.  We expect only a tiny fraction of the
# flux for a psf like object to be outside this radius for DES.  Would need to
# tune this for a different survey

RADIUS = 10

# we use maxrad of RADIUS for calculating the sums, makes the aperture circular
DES_PIXEL_SCALE = 0.263
MAXRAD = RADIUS*DES_PIXEL_SCALE

MAGZP = 30.0
