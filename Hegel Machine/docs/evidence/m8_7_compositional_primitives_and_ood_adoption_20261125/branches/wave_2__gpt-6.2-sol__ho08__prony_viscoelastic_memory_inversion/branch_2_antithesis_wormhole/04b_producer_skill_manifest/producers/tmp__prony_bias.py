import pickle,numpy as np
from scipy.optimize import least_squares
from prony_probe import basis,solve
p=pickle.load(open('/holdout_task/public_rheology.pkl','rb')); x=p['experiment']; t=x['train_t']; e=x['train_strain']; y=x['train_stress_raw']; tt=x['test_t']; et=x['test_strain']; d=np.column_stack((np.ones_like(t),(t-t.mean())/np.ptp(t))); grid=np.geomspace(.02,20,100); a=basis(t,e,grid,'ramp'); clean=p['gt_test_stress']
c,b,r,s=solve(a,d,y); good=abs(r-np.median(r))<6*s
lo=np.r_[np.zeros(4),[-np.inf]*2,[np.log(.02)]*3]; hi=np.r_[[np.inf]*6,[np.log(20)]*3]
v0=np.r_[c[0],25,15,10,b,np.log([.2,1.5,10])]; a0=lambda v,T,E:basis(T,E,np.exp(v[6:]),'ramp')
def pred(v):return a0(v,t,e)@v[:4]+d@v[4:6]
def fit(obs,start,mask=good,g=None):
 if g is None: return least_squares(lambda v:(pred(v)-obs)[mask],start,bounds=(lo,hi),max_nfev=100,ftol=1e-9)
 return least_squares(lambda q:(pred(np.r_[g,q])-obs)[mask],start[1:],bounds=(lo[1:],hi[1:]),max_nfev=100,ftol=1e-9)
f=fit(y,v0); v=f.x; mu=pred(v); res=(y-mu)[good]; var=np.dot(res,res)/(len(res)-len(v)); print('fit G',v[0],'tau',np.exp(v[6:]),'G linear SE',np.sqrt(var*np.linalg.pinv(f.jac.T@f.jac)[0,0]),'res lag 1..5',[round(np.corrcoef(res[k:],res[:-k])[0,1],3) for k in range(1,6)])
for g in np.arange(9,15.01,.5):
 f0=fit(y,v,g=g); w=np.r_[g,f0.x]; print('profile',g,'delta chi2',round((2*f0.cost-np.dot(res,res))/var,3),'tau',np.exp(w[6:]).round(3),'test err',round(np.linalg.norm(a0(w,tt,et)@w[:4]-clean)/np.linalg.norm(clean),5))
rng=np.random.default_rng(4829)
for g in [v[0],8.,12.5,17.]:
 if g==v[0]: w=v
 else: w=np.r_[g,fit(y,v,g=g).x]
 sim=np.array([fit(pred(w)+rng.normal(0,np.sqrt(var),len(t)),w).x for j in range(100)])
 print('parametric recovery, simulated G',round(g,4),'mean bias',round(np.mean(sim[:,0]-g),4),'center/5%/95%',np.quantile(sim[:,0],[.5,.05,.95]).round(3),'long tau median',round(np.median(np.exp(sim[:,-1])),3),'G raw fail fraction',round(np.mean(abs(sim[:,0]-g)>.06*g),3))
