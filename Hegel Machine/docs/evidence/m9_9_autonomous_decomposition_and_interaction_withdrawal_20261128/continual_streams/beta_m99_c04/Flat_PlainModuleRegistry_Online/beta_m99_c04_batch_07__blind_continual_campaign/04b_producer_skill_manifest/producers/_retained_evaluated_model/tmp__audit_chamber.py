import itertools, runpy, sys
import numpy as np
sys.path.insert(0,'/holdout_task')
# Reuse the public-data setup, not the earlier fold loop.
p='/tmp/compare_chamber.py'
exec(compile(open(p).read().split("for x,z in itertools.product('PRO',repeat=2):")[0],p,'exec'))
rows=np.arange(len(y)); shift=rows//nsen; i,j=s['sensor_ij'].T
folds={'rosette':(rows%nsen)//4,'shift':shift,'wall':np.tile((i==0)|(i==nx-1)|(j==0)|(j==ny-1),len(s['mu_shifts']))}
for x,z in itertools.product('PRO',repeat=2):
    L=kron(eye(ny),axis('x',x),format='csc')+kron(axis('y',z),eye(nx),format='csc')+V
    S=np.vstack([splu(L+mu*I).solve(rhs)[ix+nx*iy] for mu in s['mu_shifts']]); result=[]
    for groups in folds.values():
        scores=[]
        for group in np.unique(groups):
            test=groups==group
            if test.all(): continue
            err=(fit(S,~test,sr)-y)[test]
            scores.append(np.sqrt(np.mean(err**2)))
        result.append(max(scores) if scores else None)
    print(x+z,'worst RMSE (held-out rosette, shift, wall/interior):',result)
