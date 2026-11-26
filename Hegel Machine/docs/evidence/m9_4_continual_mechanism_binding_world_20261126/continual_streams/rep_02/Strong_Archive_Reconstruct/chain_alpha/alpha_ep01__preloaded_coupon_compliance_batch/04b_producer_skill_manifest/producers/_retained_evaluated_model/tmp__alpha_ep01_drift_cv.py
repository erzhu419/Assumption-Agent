import pickle, sys, numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0,'/holdout_task')
import alpha_ep01_solver as s
d=pickle.load(open('/holdout_task/public_alpha_ep01.pkl','rb'))[0]['dataset']
a,b,y,c=s._prepare(d); c=c[:len(y)]; t=d['eval_times']; tr=d['train']; lab=tr['lab_id']; scale=np.maximum(np.abs(a.hold),.5)
L=np.unique(lab)
# Learn a drift gain exclusively from inner held-out residuals.
cache={}
def residual(train_labs, eval_lab):
    key=tuple(train_labs)
    if key not in cache:
        tau,w=s._fit(a,y,np.isin(lab,key)&c)
        cache[key]=y-a.matrix(np.log(tau))@w
    return cache[key][lab==eval_lab]
def estimate(r, missing, robust):
    r=r[missing]; rr=r[:,-1]; z=r[:,:-1]
    if robust:
        return minimize_scalar(lambda g: np.log1p(((z-g*rr[:,None]*t)/(.018+.006*t))**2).mean(),bounds=(0,1),method='bounded').x
    return np.clip(np.sum(z*rr[:,None]*t)/np.sum((rr[:,None]*t)**2),0,1)
raw=np.empty_like(y); gains={key:np.full(len(y),np.nan) for key in ('archive','zero_full','full','nested_robust','nested_l2')};
for j in L:
    outer=L[L!=j]; idx=lab==j; raw[idx]=residual(outer,j)
    rr=np.concatenate([residual(outer[outer!=k],k) for k in outer]); cc=np.concatenate([c[lab==k] for k in outer]);
    for key,robust in [('nested_robust',True),('nested_l2',False)]:
        g=[estimate(rr,cc,robust),estimate(rr,~cc,robust)]; print('lab',j,key,np.round(g,3)); gains[key][idx]=np.where(c[idx],*g)
    gains['archive'][idx]=np.where(c[idx],.25,1.); gains['zero_full'][idx]=np.where(c[idx],0.,1.); gains['full'][idx]=1.
for k,g in gains.items():
    e=raw[:,:-1]-g[:,None]*raw[:,-1,None]*t
    def summary(m):
        abs_e=np.abs(e[m]); sec=np.abs(e[m,-1]/scale[m]); return np.array([np.sqrt(np.mean(abs_e**2)),np.median(abs_e),np.quantile(abs_e,.9),np.median(sec),np.quantile(sec,.9)])
    slices=[lab==j for j in L]+[tr['stratum']==j for j in np.unique(tr['stratum'])]+[c,~c,np.abs(a.hold)<.5,np.linalg.norm(tr['initial_memory_states'],axis=1)>np.quantile(np.linalg.norm(tr['initial_memory_states'],axis=1),.9)]
    print(k,'[curve RMSE/medAE/p90, secant medAE/p90] worst',np.round(np.max([summary(m) for m in slices],axis=0),5),'complete',np.round(summary(c),5),'missing',np.round(summary(~c),5))
for m in (c,~c):
    print('raw rate abs and normalized terminal residual abs q50/80/90/95/99',np.round(np.quantile(np.abs(raw[m][:,[-1,-2]])/[1,t[-1]], [.5,.8,.9,.95,.99],axis=0),4))
