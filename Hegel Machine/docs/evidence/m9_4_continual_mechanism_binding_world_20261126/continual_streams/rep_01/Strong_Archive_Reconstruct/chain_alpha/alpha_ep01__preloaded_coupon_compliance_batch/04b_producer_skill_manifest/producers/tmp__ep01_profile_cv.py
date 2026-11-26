import pickle
import numpy as np
from scipy.optimize import least_squares
from alpha_ep01_solver import _aligned, _physical, _Design, _fit, _mad, _crossing

case = pickle.load(open('/holdout_task/public_alpha_ep01.pkl','rb'))[0]
d, g = case['dataset'], case['gt']
tr, te = d['train'], d['test']
lab = tr['lab_id']; n = len(lab); t = d['eval_times']
x, c = _aligned(d)
a, b = [_Design(*_physical(d,p),d['past_dt'],t,f,p['initial_memory_states']) for p,f in [(tr,x[:n]),(te,x[n:])]]
y = np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]
v = np.r_[t,1.]
bounds = (np.r_[np.log([.051]*2),[-np.inf]*7],np.r_[np.log([24.9]*2),[np.inf]*7])
def rmse(e): return np.sqrt(np.mean(e**2))
for l in [*np.unique(lab),-1]:
    rows=lab!=l; complete=rows&c[:n]; lt,w=_fit(a,y,complete)
    r=a.matrix(lt)@w-y
    scale=np.r_[np.full(len(t),_mad(r[complete,:-1].ravel())),_mad(r[complete,-1])]
    u=v/scale; u=u/np.linalg.norm(u)
    for mode in ('soft_l1','cauchy','profile'):
        if mode=='soft_l1': p=np.r_[lt,w]
        else:
            selected=complete if mode=='cauchy' else rows
            missing=~c[:n][selected]
            def fun(p):
                e=(a.matrix(p[:2])[selected]@p[2:]-y[selected])/scale
                e[missing]-=(e[missing]@u)[:,None]*u
                return e.ravel()
            def jac(p):
                z,dz=a.matrix(p[:2],True)
                j=np.concatenate((dz[selected]*p[2:4],z[selected]),axis=2)/scale[None,:,None]
                j[missing]-=u[None,:,None]*np.einsum('h,nhp->np',u,j[missing])[:,None,:]
                return j.reshape(-1,9)
            p=least_squares(fun,np.r_[lt,w],jac=jac,loss='cauchy',f_scale=2.5,x_scale='jac',bounds=bounds,max_nfev=80).x
        if l!=-1:
            held=~rows; r=y[held]-a.matrix(p[:2])[held]@p[2:]
            gain=np.where(c[:n][held],.15,1); e=r[:,:-1]-(gain*r[:,-1])[:,None]*t
            # Endpoint proxy uses several late horizons to suppress point noise.
            end=np.median(e[:,-5:]/t[-5:],axis=1)*t[-1]
            q=c[:n][held]
            print(l,mode,'curve MAD; endpoint median abs (complete/missing); rate MAD',np.round([_mad(e.ravel()),np.median(abs(end[q])),np.median(abs(end[~q])),_mad(r[:,-1])],6))
        else:
            z=b.matrix(p[:2])@p[2:]; i=te['strain_0']; f=i[:,None]+z[:,:-1]+((te['instant_rate_obs']-z[:,-1])*np.where(c[n:],.15,1))[:,None]*t
            err=[np.exp(p[:2])-g['taus'], f-g['te_clean'], (f[:,-1]-i)/np.maximum(abs(b.hold),.5)-g['te_secant'], _crossing(t,i,f,te['critical_strain'])-g['te_cross']]
            print(mode,'public RMSE',np.round(list(map(rmse,err)),6),'worst lab',np.round([max(rmse(e[te['lab_id']==j]) for j in np.unique(te['lab_id'])) for e in err[1:]],6))
