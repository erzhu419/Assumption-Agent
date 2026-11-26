import copy, importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_ep01_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
for case in pickle.load(open('/holdout_task/public_alpha_ep01.pkl','rb')):
    d = case['dataset']; tr = d['train']; t = d['eval_times']; dt = d['past_dt']; lab = tr['lab_id']; n = len(lab)
    x, complete = s._align(d); x=x[:n]; complete=complete[:n]
    gain,bias=s._calibrate(d['calibration_load_cells']); design=s._Design(tr,x,gain,bias,dt,t)
    obs=tr['strain_curves_obs']; y=np.c_[obs-tr['strain_0'][:,None],tr['instant_rate_obs']]
    def fit(mask):
        rows=mask & complete
        return s._fit(s._Design(tr,x,gain,bias,dt,t,rows),y[rows])
    def base(par):
        return s._predict(design,*par,tr,complete,t)
    def grammar(par):
        tau=np.exp(par[0]); state=design.initial.copy()
        for stress, step in zip(design.past.T, dt):
            state += -np.expm1(-step/tau)*(stress[:,None]-state)
        u,v,w,q=x.T; h=design.hold; a,b=(h[:,None]-state).T
        return {
            'none': np.empty((n,0)),
            'pair_and_triple': np.column_stack([u*v,u*w,u*q,v*w,v*q,w*q,u*v*w,u*v*q,u*w*q,v*w*q]),
            'rational_contrasts': np.column_stack([(x[:,i]+sign*x[:,j])/(1+alpha*(x[:,i]-sign*x[:,j])**2) for i in range(4) for j in range(i+1,4) for sign in [-1,1] for alpha in [.25,1.]]),
            'stress_and_memory': np.column_stack([h*h,h*abs(h),a*a,b*b,a*b,(a-b)/(1+(a+b)**2),h/(1+h*h)]),
            'within_lab': np.column_stack([x-np.vstack([np.mean(x[lab==l],axis=0) for l in lab]), h-np.array([np.mean(h[lab==l]) for l in lab])]),
        }
    def drift_fit(z, res, rows):
        if z.shape[1]==0: return np.zeros(n)
        z=z-z[rows].mean(axis=0); z=z/np.maximum(z[rows].std(axis=0),1e-8)
        z=z.clip(-5,5); zz=z[rows]; yy=res[rows]; lam=0.005
        p=least_squares(lambda p: np.r_[zz@p-yy,lam*p],np.zeros(z.shape[1]),loss='soft_l1',f_scale=.006).x
        return z@p
    def score(pred,rows):
        e=np.abs(pred[rows]-obs[rows]); hold=np.maximum(np.abs(design.hold[rows]),.5)
        return np.array([np.minimum(e,.1).mean(), np.median(e[:,-1]/hold)])
    worst={}; selected=[]
    for l in np.unique(lab):
        outer=lab!=l; par=fit(outer); p=base(par); z=grammar(par)
        inner_res=np.zeros(n); inners=[]
        for j in np.unique(lab[outer]):
            inner=outer & (lab!=j); val=lab==j; ipar=fit(inner); ip=base(ipar); iz=grammar(ipar)
            r=np.median((obs[:,4:]-ip[:,4:])/t[4:],axis=1)
            inner_res[val]=r[val]; inners.append((inner,val,ip,iz,r))
        inner_scores={name:np.mean([score(ip+drift_fit(iz[name],r,inner & complete)[:,None]*t,val & complete)[0] for inner,val,ip,iz,r in inners]) for name in z}
        selected.append(min(inner_scores,key=inner_scores.get))
        for name,zz in z.items():
            pred=p+drift_fit(zz,inner_res,outer & complete)[:,None]*t
            slices=[score(pred,(lab==l)&m) for m in [complete,~complete] if np.any((lab==l)&m)]
            line=np.max(slices,axis=0); worst[name]=np.maximum(worst.get(name,0),line)
        raw=y[:,-1]-design.matrix(par[0])[:,-1]@par[1]
        print('lab',int(l),'complete rate residual median',round(float(np.median(raw[(lab==l)&complete])),6),'inner selection',selected[-1])
    print('worst lab/completeness [capped strain MAE, median secant AE]',{k:v.round(6).tolist() for k,v in worst.items()})
    d2=copy.deepcopy(d); factors=np.array([.58,1.41,.83,1.22]); offsets=np.array([.65,-.4,.89,-1.1]); cal=d2['calibration_load_cells']; l=cal['lab_id']; cal['cmd_stress']=factors[l]*cal['cmd_stress']+offsets[l]
    for name in ('train','test'):
        p=d2[name]; l=p['lab_id']; p['past_stress_cmd']=factors[l,None]*p['past_stress_cmd']+offsets[l,None]; p['stress_hold_cmd']=factors[l]*p['stress_hold_cmd']+offsets[l]
    a,b=s.solve_alpha_ep01(d),s.solve_alpha_ep01(d2)
    differences={k:float(np.max(np.abs(a[k]-b[k]))) for k in a}; print('load-cell recoding max differences',differences)
    for k in a: np.testing.assert_allclose(a[k],b[k],atol=2e-7,rtol=2e-7)
