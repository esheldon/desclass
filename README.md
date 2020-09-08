# desclass

Probabilistic star galaxy classifier

# examples

```
desclass \
    --flist DES0426-4040_r4939p01_g_meds-Y6A1.fits.fz \
            DES0426-4040_r4939p01_r_meds-Y6A1.fits.fz \
            DES0426-4040_r4939p01_i_meds-Y6A1.fits.fz \
            DES0426-4040_r4939p01_z_meds-Y6A1.fits.fz \
    --stars hsc/opt-griz-hsc-stars.fits \
    --output test.fits \
    --start 100 --end 200 \
    --seed 5
```
# dependencies

- numpy
- numba
- ngmix
- meds
- esutil
- fitsio
- hickory and matplotlib (optional for plotting)
