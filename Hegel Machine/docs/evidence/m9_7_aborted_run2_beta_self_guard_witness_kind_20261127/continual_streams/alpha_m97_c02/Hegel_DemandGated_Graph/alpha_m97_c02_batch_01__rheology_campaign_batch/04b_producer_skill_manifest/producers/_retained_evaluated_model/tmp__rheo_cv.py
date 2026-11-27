import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
import alpha_m97_c02_batch_01_solver as m
d=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_01.pkl','rb'))[0]['dataset']
dt,t,(b,_)=m._prepare(d); s=d['train']; y=s['strain_curves_obs']-s['strain_0'][:,None]; r=s['instant_rate_obs']; labs=s['lab_id']; active=b[-1]>0
rm=lambda x: np.sqrt(np.mean(x*x))
for q, log in [(0.,True),(.25,True),(1.,True),(4.,True),(1.,False)]:
 ec=np.empty_like(y); er=np.empty_like(r); est=[]
 for lab in np.unique(labs):
  te=labs==lab; tr=~te
  tau,w=m._fit(dt,t,tuple(x[tr] for x in b),y[tr],r[tr],q,log); est.append(np.round(tau,5).tolist())
  c,a=m._basis(tau,dt,t,tuple(x[te] for x in b),log); ec[te]=c@w-y[te]; er[te]=a@w-r[te]
 print('rate_weight,log',q,log,'taus',est,'pooled',rm(ec),rm(er),'worst lab/activation',np.max([[rm(ec[(labs==lab)&(active==on)]),rm(er[(labs==lab)&(active==on)])] for lab in np.unique(labs) for on in (False,True)],axis=0),'worst lab/horizon',max(rm(ec[labs==lab,k]) for lab in np.unique(labs) for k in range(len(t))))
print('all-data parameters',m._fit(dt,t,b,y,r))
