exec(open('/tmp/alpha02_cv.py').read().split("for loss in ('cauchy'")[0])
from itertools import combinations
from scipy.optimize import least_squares
import copy
X=state[2][:,1:]; hold=state[1]
# Each frozen challenger is fitted only on the other labs. Complete rows
# isolate model structure from the prescribed missing-feature imputation.
pairs=list(combinations(range(X.shape[1]),2))
phi=np.column_stack([X[:,i]*X[:,j] for i,j in pairs]+[hold/(1+hold**2)])
e=np.empty((2,*y.shape)); f=np.empty((2,len(r)))
for j in np.unique(lab):
    val=lab==j; fit=~val; clean=fit&~missing
    p,w=m._fit(tuple(a[fit] for a in state),dt,time,y[fit],r[fit])
    A,B=m._design(p,state,dt,time)
    ec=A@w-y; er=B@w-r
    cs=m._scale(ec[clean]); rs=m._scale(er[clean])
    # Products test exchange/nonadditivity; the bounded rational term
    # tests stress-dependent drift omitted by the specified linear flow.
    D=phi[:,None,:]*time[None,:,None]
    c=least_squares(lambda c: np.r_[((ec[clean]+D[clean]@c)/cs).ravel(),(er[clean]+phi[clean]@c)/rs], np.zeros(phi.shape[1]), loss='cauchy', f_scale=2.5).x
    e[:,val]=[ec[val],ec[val]+D[val]@c]; f[:,val]=[er[val],er[val]+phi[val]@c]
for v in range(2):
    print(('specified','augmented')[v])
    for label, subset in [('complete',~missing),('imputed',missing)]:
        print(label,'median by held-out lab: (worst strain horizon, initial rate, signed rate)',[(int(j),round(np.median(abs(e[v,(lab==j)&subset]),axis=0).max(),5),round(np.median(abs(f[v,(lab==j)&subset])),5),round(np.median(f[v,(lab==j)&subset]),5)) for j in np.unique(lab)])
# Explicit nuisance check: perturb uncoupled telemetry, not calibration or labels.
d2=copy.deepcopy(d)
for split in ('train','test'):
    d2[split]['aux_chamber_telemetry'] *= -100
for a,b in zip(m._prepare(d),m._prepare(d2)):
    np.testing.assert_array_equal(a,b)
print('telemetry perturbation leaves all model inputs unchanged')
