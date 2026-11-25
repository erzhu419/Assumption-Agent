import contextlib, io, runpy
import numpy as np

def alternative(original, mix):
    def shrink(estimates, noise):
        fits = original(estimates, noise)
        adjusted = []
        for x, v, fit in zip(estimates, noise, fits):
            if len(x) < 2:
                adjusted.append(fit)
                continue
            center = np.median(x)
            residual = x - center
            # Recover the local shrinkage fractions from the existing fit.
            fraction = np.divide(fit-center, residual, out=np.zeros_like(x), where=residual!=0)
            med_spread = np.median((fraction*v/np.maximum(1-fraction, 1e-15))[residual!=0]) if (residual!=0).any() else 0.
            weights = 1/(v+med_spread)
            shift = mix * (np.average(x, weights=weights)-center)
            adjusted.append(fit + shift*v/(v+med_spread))
        return adjusted
    return shrink

from importlib.util import spec_from_file_location, module_from_spec
src = '/tmp/assay_stress.py'
spec = spec_from_file_location('assay_stress', src)
for mix in (0., .5, 1.):
    code = open(src).read()
    marker = 'spec.loader.exec_module(assay)'
    code = code.replace(marker, marker+'\nassay._shrink = alternative(assay._shrink, mix)')
    print('pool precision mix', mix, flush=True)
    exec(compile(code, src, 'exec'), {'alternative': alternative, 'mix': mix})
