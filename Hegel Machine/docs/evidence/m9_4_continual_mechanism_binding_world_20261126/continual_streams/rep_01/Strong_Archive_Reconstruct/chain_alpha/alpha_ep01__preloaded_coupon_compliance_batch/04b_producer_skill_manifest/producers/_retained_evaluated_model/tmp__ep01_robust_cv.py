import numpy as np, pickle
from scipy.optimize import least_squares
from alpha_ep01_solver import _aligned,_physical,_Design,_fit,_mad,_crossing
p=pickle.load(open('/holdout_task/public_alpha_ep01.pkl','rb'))[0]; d,g=p['dataset'],p['gt']; tr,te=d['train'],d['test']; n=len(tr['lab_id']); x,c=_aligned(d); t=d['eval_times']; labs=tr['lab_id']; a,b=(_Design(*_physical(d,q),d['past_dt'],t,f,q['initial_memory_states']) for q,f in [(tr,x[:n]),(te,x[n:])]); y=np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]
def rmse(e): return np.sqrt(np.mean(e**2))
def refine(lt,w,keep,mode):
    if mode=='baseline': return lt,w
    r=a.matrix(lt)[keep]@w-y[keep]; scale=np.r_[np.full(len(t),_mad(r[:,:-1].ravel())),_mad(r[:,-1])]
    if mode=='row':
        # A coherent shift across most time samples is not the linear drift.
        shift=np.median(r[:,:-1],axis=1)
        keep=keep.copy(); keep[keep]=np.abs(shift-np.median(shift))<8*_mad(shift)
    def fun(p): return ((a.matrix(p[:2])[keep]@p[2:]-y[keep])/scale).ravel()
    def jac(p):
        z,dz=a.matrix(p[:2],True); return (np.concatenate((dz[keep]*p[2:4],z[keep]),axis=2)/scale[None,:,None]).reshape(-1,9)
    f=least_squares(fun,np.r_[lt,w],jac=jac,loss='cauchy',f_scale=2.5,x_scale='jac',bounds=(np.r_[np.log([.051]*2),[-np.inf]*7],np.r_[np.log([24.9]*2),[np.inf]*7]),max_nfev=100)
    return f.x[:2],f.x[2:]
for l in [-1,*np.unique(labs)]:
    keep=c[:n]&(labs!=l); lt,w=_fit(a,y,keep)
    for mode in ('baseline','cauchy','row'):
        ll,ww=refine(lt,w,keep,mode); base=a.matrix(ll)@ww; rr=y-base; delta=np.where(c[:n],.15,1)*rr[:,-1]; proxy=(np.median(rr[:,-6:-1]/t[-5:],axis=1)-delta)*t[-1]
        if l!=-1:
            print(l,mode,'rate MAD, curve MAD, endpoint proxy median abs complete/missing:',np.round([_mad(rr[labs==l,-1]),_mad((rr[labs==l,:-1]-delta[labs==l,None]*t).ravel()),np.median(abs(proxy[(labs==l)&c[:n]])),np.median(abs(proxy[(labs==l)&~c[:n]]))],6)); continue
        pred=b.matrix(ll)@ww; initial=te['strain_0']; curve=initial[:,None]+pred[:,:-1]+((te['instant_rate_obs']-pred[:,-1])*np.where(c[n:],.15,1))[:,None]*t
        es=[curve-g['te_clean'],(curve[:,-1]-initial)/np.maximum(np.abs(b.hold),.5)-g['te_secant'],_crossing(t,initial,curve,te['critical_strain'])-g['te_cross']]
        print(mode,'tau',np.exp(ll),'overall',np.round(list(map(rmse,es)),6),'worst lab',np.round([max(rmse(e[te['lab_id']==i]) for i in np.unique(te['lab_id'])) for e in es],6),'missing',np.round([rmse(e[~c[n:]]) for e in es],6))
