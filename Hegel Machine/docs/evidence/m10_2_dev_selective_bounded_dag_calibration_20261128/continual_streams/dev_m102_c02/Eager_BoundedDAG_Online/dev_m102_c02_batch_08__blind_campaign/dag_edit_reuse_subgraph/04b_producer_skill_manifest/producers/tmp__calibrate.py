import numpy as np,pickle,sys,importlib.util
from scipy.optimize import least_squares
from scipy.special import expit
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def load(j):
 spec=importlib.util.spec_from_file_location('f'+str(j),f'/holdout_task/_frozen_nodes/node_0{j}.py'); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod.run_node
f,g=load(1),load(2)
def abasis(u,j,mode):
 a,b=u[:,j],u[:,(j+1)%4]; d=[np.ones(len(u)),a,b]
 if mode!='affine':d+=[a*b]
 if mode in ('cubic','both','full'):d +=[a**3]
 if mode in ('both','full'):d +=[b**3]
 if mode=='full':d +=[a*a,b*b,a*a*b,a*b*b]
 return np.column_stack(d)
u,v=s['panel_a']['u'],s['panel_a']['v'];rng=np.random.default_rng(2)
bs={}; us={}; xs=np.r_[s['train']['x'],s['test']['x']]
for mode in ['affine','cyclic','cubic','both','full']:
 a=np.stack([abasis(u,j,mode) for j in range(4)]); k=a.shape[-1]; ids=np.argsort(rng.random((2500,len(u))),axis=1)[:,:k+1]; preds=np.stack([a[j]@(np.linalg.pinv(a[j][ids])@v[ids,j,None])[...,0].T for j in range(4)],axis=-1); err=np.linalg.norm(preds-v[:,None,:],axis=-1); idx=np.argmin(np.mean(np.partition(err,38,axis=0)[:39]**2,axis=0)); keep=err[:,idx]<0.8
 p=np.stack([np.linalg.lstsq(a[j,keep],v[keep,j],rcond=None)[0] for j in range(4)]); bs[mode]=p
 def forward(z):return np.column_stack([abasis(z,j,mode)@p[j] for j in range(4)])
 loo=np.column_stack([(v[keep,j]-a[j,keep]@p[j])/(1-np.sum(a[j,keep]*np.linalg.pinv(a[j,keep]).T,1)) for j in range(4)])
 sol=[least_squares(lambda z:forward(z[None,:])[0]-x,np.clip(x,-1.25,1.25),max_nfev=80) for x in xs];us[mode]=np.array([z.x for z in sol]); print('A',mode,keep.sum(),'LOO',np.mean(abs(loo)),np.max(abs(loo)), 'inverse',max(np.linalg.norm(z.fun) for z in sol),np.max(abs(us[mode])), '\n',p.round(5))
u,v=s['panel_b']['u'],s['panel_b']['v']
def aff(x):return np.c_[np.ones(len(x)),x]
def loo(a,v):
 p=np.linalg.lstsq(a,v,rcond=None)[0];r=(a@p-v)/(1-np.sum(a*np.linalg.pinv(a).T,axis=1))[:,None];return np.mean(abs(r)),np.max(abs(r))
trans={ 'linear':lambda z:aff(z),'cubic':lambda z:np.c_[aff(z),z**3], 'quintic':lambda z:np.c_[aff(z),z**3,z**5] }
for sc in [.3,.6,1.,1.3,1.6,2.,2.5]:
 for name,fun in [('sin',np.sin),('tanh',np.tanh),('atan',np.arctan),('sinh',np.sinh)]:
  trans[name+str(sc)]=lambda z,sc=sc,fun=fun:aff(fun(sc*z))
  trans[name+'lin'+str(sc)]=lambda z,sc=sc,fun=fun:np.c_[aff(z),fun(sc*z)]
print('B LOO',sorted((loo(t(u),v),n) for n,t in trans.items())[:16]); print('B cubic coeff',np.linalg.lstsq(trans['cubic'](u),v,rcond=None)[0].round(4))
for mode,z in us.items():
 scores=[]
 for n,t in trans.items():
  c=t(z)@np.linalg.lstsq(t(u),v,rcond=None)[0];st=dict(s,train={k:w[:52] for k,w in s['train'].items()},test={'x':s['train']['x'][52:]});r=f({'__input__':st,'node_00':{'c':c[:52],'query':c[52:72]}}); tr=g({'node_01':r})['state_pred']; ym=np.mean(abs(r['y_hat']-s['train']['y'][52:]));tm=np.mean(abs(tr-s['train']['traj'][52:])); scores.append((round(max(ym,tm),5),n,round(ym,5),round(tm,5)))
 print('CV',mode,sorted(scores)[:10])
