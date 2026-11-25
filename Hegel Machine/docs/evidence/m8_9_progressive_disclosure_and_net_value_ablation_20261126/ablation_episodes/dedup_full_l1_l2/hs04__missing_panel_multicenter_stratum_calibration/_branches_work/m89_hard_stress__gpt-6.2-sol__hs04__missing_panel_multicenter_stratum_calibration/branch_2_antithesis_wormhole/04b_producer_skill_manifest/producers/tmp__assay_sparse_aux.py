import numpy as np


def with_auxiliary(assay, mode):
    original = assay.calibrate_multicenter_assay
    base_affines = assay._reference_affines
    def calibrate(dataset):
        ref = dataset['reference_controls']
        affines, pooled = base_affines(ref)
        for part in (dataset['train'], dataset['query']):
            for c in np.unique(part['center_id']):
                if c not in affines:
                    continue
                r = ref['X'][ref['center_id'] == c]
                counts = np.isfinite(r).sum(0)
                supported = np.array([np.ptp(v[np.isfinite(v)]) > 0 if n >= 2 else False for v,n in zip(r.T,counts)])
                raw = part['X'][part['center_id'] == c]
                observed = np.isfinite(raw).sum(0)
                if not (supported & (observed >= 20)).any():
                    continue
                loc, scale = (v.copy() for v in affines[c])
                mu = np.divide(np.nansum(raw,axis=0), observed, out=loc.copy(), where=observed>0)
                sd = np.sqrt(np.divide(np.nansum((raw-mu)**2,axis=0), observed-1, out=scale**2, where=observed>1))
                known = supported & (observed >= 20) & (sd > 0)
                if not known.any():
                    continue
                pop_mean = np.median(((mu-loc)/scale)[known])
                pop_scale = np.median((sd/scale)[known])
                unknown = (~supported) & (observed >= 20) & (sd > 0)
                rough = sd[unknown] / pop_scale
                scale[unknown] = np.sqrt(scale[unknown]*rough)
                if mode == 'location_and_scale':
                    loc[unknown] = .5 * (loc[unknown] + mu[unknown] - pop_mean*scale[unknown])
                affines[c] = loc, scale
        assay._reference_affines = lambda _: (affines, pooled)
        try:
            return original(dataset)
        finally:
            assay._reference_affines = base_affines
    return calibrate

src = '/tmp/assay_stress.py'
text = open(src).read()
marker = 'spec.loader.exec_module(assay)'
for mode in ('scale', 'location_and_scale'):
    print('sparse fallback',mode,flush=True)
    trial = text.replace(marker,marker+'\nassay.calibrate_multicenter_assay = with_auxiliary(assay, mode)')
    exec(compile(trial,src,'exec'), {'with_auxiliary':with_auxiliary,'mode':mode})
