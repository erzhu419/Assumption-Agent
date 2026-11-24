import contextlib, io, os
from pathlib import Path
from time import perf_counter
import numpy as np
from peclet_tail_prior import invert_point_source
with contextlib.redirect_stdout(io.StringIO()):
    ns={}
    exec(Path('/tmp/peclet_compare.py').read_text().split('for k in (0,1,3,6):')[0],ns)
for knee, multiplier in [(1,12),(2,12),(2,25),(3,25)]:
    os.environ.update(PECLET_KNEE_TEST=str(knee),PECLET_PRIOR_TEST=str(multiplier))
    for name, cases in ns['sets'].items():
        errors=[]
        for p,xy,q in cases:
            start=perf_counter(); r=invert_point_source(p)
            errors.append((np.linalg.norm(r['source_xy']-xy),abs(r['strength_Q']/q-1),perf_counter()-start))
        e=np.asarray(errors); fail=np.flatnonzero((e[:,0]>.025)|(e[:,1]>.08))
        print(knee,multiplier,name,'max',e.max(0).round(5),'failed',fail.tolist(),'p90',np.quantile(e,.9,axis=0).round(5),flush=True)
        if name=='new': print('failed errors',e[fail,:2].round(4).tolist(),flush=True)
