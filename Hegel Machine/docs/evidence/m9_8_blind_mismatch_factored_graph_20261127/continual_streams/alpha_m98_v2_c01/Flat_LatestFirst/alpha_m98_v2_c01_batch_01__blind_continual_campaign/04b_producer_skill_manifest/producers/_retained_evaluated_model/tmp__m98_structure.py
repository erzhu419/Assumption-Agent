import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
    exec(open('/tmp/m98_probe.py').read(),globals())
from itertools import permutations
xtr,xte=m._features(tr,te)
o,g=m._axial_load_map(d['calibration_load_cells'])
p,h,X=spec
K=len(t); n=len(h)
A=np.r_[np.c_[-np.expm1(-t[:,None]/tau),t],np.r_[1/tau,1][None,:]]
Y=np.c_[y,v]
print('bridge',np.round(o,4),np.round(g,4)); print('lab/stratum counts',[(np.bincount(z['lab_id']),np.bincount(z['stratum'])) for z in (tr,te,dg)])
print('sweeps summary',[(k,np.round(vv[:8],4)) for k,vv in d['calibration_load_cells']['single_channel_sweeps'].items()]); print('telemetry corr with stress/X',np.corrcoef(np.c_[te['aux_chamber_telemetry'],m._specimen(te,xte,(o,g))[0:2][0],xte].T)[:4].round(2));print('diagnostic p,h mean/sd',np.c_[p,h].mean(0).round(3),np.c_[p,h].std(0).round(3))
base_state=(h[:,None]-r*tau)
for b in range(2):
 u=np.exp(-dt/tau[b]); print('branch',b,'benchtop coefficients',np.r_[-(1-u)*np.r_[np.cumprod(u[:0:-1])[::-1],1],1]*w[b])

def check(Z, name, detailed=False):
    blocks=[z[:,None,:]*A[None,:,j,None] for j,z in enumerate(Z)]; B=np.concatenate(blocks,2); folds=[]
    def fit(ix):return np.linalg.lstsq(B[ix].reshape(-1,B.shape[-1]),Y[ix].ravel(),rcond=None)[0]
    coef=fit(np.arange(n)); E=B@coef-Y
    for j in range(n): folds.append(B[j]@fit(np.flatnonzero(np.arange(n)!=j))-Y[j])
    F=np.array(folds)
    print(name,'in / LOOCV (curve,rate,max coupon curve)',*[np.round([np.sqrt(np.mean(e[:,:K]**2)),np.sqrt(np.mean(e[:,K]**2)),max(np.sqrt(np.mean(e[:,:K]**2,axis=1)))],5) for e in (E,F)])
    if detailed:
        cuts=np.r_[0,np.cumsum([z.shape[1] for z in Z])];print('coefs',[coef[a:b].round(5).tolist() for a,b in zip(cuts[:-1],cuts[1:])])
    return E,F
one=np.ones((n,1)); P=np.c_[p,h]; D=(X@w[2:])[:,None]; Z0=one[:,:0]; H=h[:,None]; G=r*tau*w[:2]
check([G[:,0:1],G[:,1:2],D],'fixed-memory rescale');check([np.c_[one,H,G[:,0]],np.c_[one,H,G[:,1]],D],'fixed-memory hold+offset');check([P,P,Z0],'linear history, no drift',True);check([np.c_[P,one],np.c_[P,one],Z0],'linear history with offset, no drift',True);check([P,P,np.c_[one,D]],'linear history and drift',True);check([P,P,X],'linear history and all drift'); check([np.c_[P,D],np.c_[P,D],Z0],'linear history and drift transferred to branches',True)
# Fit one physical hold-minus-history contrast per branch, testing shared recurrence schedules.
for nm,pp,dd in [('forward',p,dt),('reverse',p[:,::-1],dt[::-1]),('reverse stress',p[:,::-1],dt),('flat dt',p,np.full(6,dt.mean()))]:
    c,rr=m._branches((pp,h,X),dd,t,tau); gg=rr*tau*w[:2];check([gg[:,0:1],gg[:,1:2],Z0],nm+' rescale no drift',True)
