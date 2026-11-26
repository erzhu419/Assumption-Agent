exec(open('/tmp/w03_b04_eval.py').read().split('pred=np.empty_like')[0])
from scipy.optimize import least_squares
p=z[:n]; hold=h[:n]; xx=x[:n]; delta=obs-start[:,None]; k=len(t)
ov=np.maximum(abs(hold)-y,0.)**2*np.sign(hold)
work=(np.maximum(abs(p)-y,0.)**2@dt)*hold
pair=xx[:,1]-xx[:,2]
triple=xx[:,1]*xx[:,2]*xx[:,3]

def fit(a,b,w):
    q=a/w[:,None]; target=b/w
    s=np.maximum(np.linalg.norm(q,axis=0),1e-12)
    q=q/s
    init=np.linalg.lstsq(q,target,rcond=None)[0]
    return least_squares(lambda v:q@v-target,init,jac=lambda v:q,loss='cauchy',f_scale=3.,max_nfev=100).x/s

for wing in (0,1):
    i=(lab//2!=wing)&~missing; j=(lab//2==wing)&~missing
    tau,coef=m._fit(xx[i],hold[i],p[i],dt,t,y,delta[i],rate[i])
    c,r=m._design(tau,xx,hold,p,dt,t,y)
    state=np.zeros((n,2))
    for s,duration in zip(p.T,dt):
        f=np.exp(-duration/tau)
        state=state*f+s[:,None]*(1-f)
    shape=-np.expm1(-t[:,None]/tau)
    independent=((hold[:,None]-state)[:,None,:]*shape)@coef[:2]
    independent+=(xx@coef[2:7]+coef[7]*ov)[:,None]*t
    independent+=coef[8]*work[:,None]*shape[:,0]
    assert np.allclose(independent,c@coef,atol=1e-12,rtol=1e-12)
    extra=np.column_stack((pair**2,triple,ov/(1+hold**2),work/(1+hold**2)))
    ec=extra[:,None,:]*t[None,:,None]
    ec[:,:,3]=extra[:,3,None]*shape[:,0]
    er=extra.copy(); er[:,3]/=tau[0]
    a=np.concatenate((c,ec),axis=2); b=np.column_stack((r,er))
    base_err=(c@coef-delta)[i]; rate_err=(r@coef-rate)[i]
    s=[max(1e-7,1.4826*np.median(abs(v-np.median(v)))) for v in (base_err,rate_err)]
    w=np.r_[np.full(i.sum()*k,s[0]),np.full(i.sum(),s[1])]
    A=np.vstack((a[i].reshape(-1,13),b[i])); target=np.r_[delta[i].ravel(),rate[i]]
    base=fit(A[:,:9],target,w); aug=fit(A,target,w)
    for active in (False,True):
        mask=j&(yielded==active)
        if not mask.any(): continue
        errors=(c[mask]@base-delta[mask],a[mask]@aug-delta[mask],r[mask]@base-rate[mask],b[mask]@aug-rate[mask])
        print('held wing',wing,'yielded',active,'n',mask.sum(),'curve/rate median abs error, base -> expanded',*[round(np.median(abs(v)),6) for v in errors])
