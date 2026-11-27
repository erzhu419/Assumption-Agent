exec(open('/tmp/probe_ng_t05.py').read().split("print('ng_t05")[0])
from downstream_solver import _axis, solve_downstream
axes=[]
for d in 'xy':
    k=s['kappa_'+d]; v=s['vel_'+d]; h=s['L'+d]/(len(k)-1)
    axes.append(_axis(k,v,h))
A=design(*axes); B=design(axis('x','O'),axis('y','O')); M=design(wall('x',1),wall('y',0))
for penalty in (1,0):
    candidate, baseline, mixed=(np.array(scores(D,penalty)) for D in (A,B,M))
    print('penalty multiplier',penalty,'candidate',candidate,'Robin',baseline,'mixed',mixed)
    assert np.all(candidate < baseline/10), (candidate,baseline)
    assert np.all(candidate[[0,2]] < mixed[[0,2]]), (candidate,mixed)
cal=calibrate_upstream(s); out=solve_downstream(cal)
w,_=fit(A,np.ones(len(y),bool)); error=np.max(np.abs(out['weights']-w)); print('shipped downstream weight agreement',error); assert error < 1e-10
