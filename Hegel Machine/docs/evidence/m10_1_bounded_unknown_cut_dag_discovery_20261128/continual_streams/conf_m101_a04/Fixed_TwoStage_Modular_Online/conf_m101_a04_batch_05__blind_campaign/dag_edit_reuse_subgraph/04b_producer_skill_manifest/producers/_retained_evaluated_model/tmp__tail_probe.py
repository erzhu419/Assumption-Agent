import importlib.util,pickle
import numpy as np

def load(name,path):
 s=importlib.util.spec_from_file_location(name,'/holdout_task/'+path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
up=load('up','node_00.py'); dn=load('dn','_frozen_nodes/node_01.py')
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=s['train']['y']; t=s['train']['traj']; n=len(y); ids=np.arange(n)
x=up._decode(s['panel_a'],s['train']['x']); b,c1,c2=up._features(s); base=np.c_[.03*b,c1,c2][:n]; raw=s['train']['x']; groups={'raw tails':[a for col in raw.T for a in (np.argsort(col)[:12],np.argsort(col)[-12:])], 'latent tails':[a for col in x.T for a in (np.argsort(col)[:12],np.argsort(col)[-12:])], 'random':[g for seed in (408,6182) for g in np.array_split(np.random.default_rng(seed).permutation(n),5)], 'radial':list(np.array_split(np.argsort(np.linalg.norm(raw,axis=1)),5)), 'public':[ids[-20:]]}
def score(scale,center,w=1.,r=1.):
 h=base.copy(); h[:,:4]*=w; h[:,4:8]*=r
 for group,folds in groups.items():
  vals=[]
  for te in folds:
   tr=np.setdiff1d(ids,te); z=h-(h[tr].mean(0) if center else 0); z*=scale
   p=dn.run_node({'node_00':{'h':z[np.r_[tr,te]],'n_train':len(tr),'y':y[tr],'traj':t[tr]}})
   vals.append((np.mean(abs(p['y_hat']-y[te])),np.mean(abs(p['state_pred']-t[te]))))
  a=np.array(vals); yield group,np.round(a.max(0),5).tolist()
for scale in (0.6,1.,1.5,2.,3.,5.):
 for center in (False,True): print(scale,center,dict(score(scale,center)))
for w in (0.,1.,3.):
 for r in (.5,2.): print('rel',w,r,dict(score(1.,True,w,r)))
