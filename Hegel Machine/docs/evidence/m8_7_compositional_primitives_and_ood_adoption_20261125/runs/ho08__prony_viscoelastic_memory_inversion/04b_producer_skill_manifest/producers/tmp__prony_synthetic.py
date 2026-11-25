import pickle, sys
import numpy as np
from importlib import import_module
fit=import_module(sys.argv[1]).fit_prony_and_predict_stress
p=pickle.load(open('/holdout_task/public_rheology.pkl','rb')); base=p['experiment']; t=base['train_t']; e=base['train_strain']; rng=np.random.default_rng(710)
def stress(t,e,g,taus,amps):
    left=t[:,None]-t[:-1]; right=t[:,None]-t[1:]; past=right>=0
    rate=np.diff(e)/np.diff(t); out=g*e.copy()
    for tau,a in zip(taus,amps):
        out+=a*tau*((np.exp(-np.maximum(right,0)/tau)-np.exp(-np.maximum(left,0)/tau))*past)@rate
    return out
def evaluate(g,experiment,target):
    r=fit(experiment); print('err',round(np.linalg.norm(r['test_stress']-target)/np.linalg.norm(target),5),round(r['G_inf']/g-1,5),flush=True)
print('public'); evaluate(p['gt_G_inf'],base,p['gt_test_stress'])
s=base['test_t']; u=s+0.012*np.sin(7*s); v=base['test_strain'];
for g,taus,amps in [(17.,[.12,2.3],[30,13]),(9.,[.07,.85,5.],[21,16,9]),(20.,[.3,2.4,12.],[25,17,10]),(14.,[.10,.58,2.6,11.],[27,16,12,7]),(12.5,[.18,1.4,8.5],[28,16.5,9])]:
    print('spectrum',taus)
    target=stress(u,v,g,taus,amps)
    y=stress(t,e,g,taus,amps)+2.1-.08*t+rng.normal(0,.015,len(t))
    j=rng.choice(len(t),12,replace=False); y[j]+=rng.choice([-1,1],12)*rng.uniform(.35,1.8,12)
    evaluate(g,dict(base,train_stress_raw=y,test_t=u),target)
