import sys, pickle, numpy as np, importlib.util
from itertools import combinations_with_replacement
from pathlib import Path
R=Path('/holdout_task')
s=pickle.load(open(R/'public_batch.pkl','rb'))[0]['study']
def load(p):
 spec=importlib.util.spec_from_file_location(p.stem,p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
mods=[load(R/'_frozen_nodes'/f'node_{i:02}.py') for i in range(3)]
lst=lambda x,y: np.linalg.lstsq(x,y,rcond=1e-9)[0]
T=np.column_stack((s['train']['y'],s['train']['traj'])); n=len(T)
W=mods[0].run_node({'__input__':s})['calibrated'][:n]
print('calibrated ranges', W.min(0),W.max(0),'singular',np.linalg.svd(W-W.mean(0),compute_uv=False))
with np.printoptions(precision=5):
 p=s['panel_a']; z=mods[0]._align(p,p['v']); print('alignment quantiles',np.quantile(np.linalg.norm(z-p['u'],axis=1),[0,.25,.5,.75,.90,.95,.98,1])); print('time singular',np.linalg.svd(T[:,1:],compute_uv=False))
def designs(W,tr):
 z=(W-W[tr].mean(0))/np.maximum(.1,W[tr].std(0)); a=np.column_stack((np.ones(len(z)),z)); powers=[q for d in (2,3) for q in combinations_with_replacement(range(8),d)]; poly=np.column_stack([np.prod(z[:,q],axis=1) for q in powers]); return a,poly,powers

def path(a,p,y,k):
 # Orthogonalized forward selection gives scale-invariant block improvement.
 a=a.copy(); p=p.copy(); y=y.copy(); q=np.linalg.qr(a,mode='reduced')[0]; p-=q@(q.T@p); y-=q@(q.T@y); cols=[]
 for _ in range(k):
  norm=np.linalg.norm(p,axis=0); score=np.sum((p.T@y)**2,axis=1)/np.maximum(1e-20,norm**2); score[norm<1e-9]=-np.inf; j=np.argmax(score); cols.append(j); q=p[:,j]/norm[j]; p-=np.outer(q,q@p); y-=np.outer(q,q@y); p[:,j]=0
 return cols

def setup(tr,va):
 a,p,terms=designs(W,tr)
 b=mods[1]._time_basis(s['eval_times'],T[tr,1:]); q=np.linalg.qr(b)[0]; Y=np.column_stack((T[:,0],T[:,1:]@q)); z=a[:,1:]; fit=mods[2].run_node({'node_00':{'features':z[np.r_[tr,va]]},'node_01':{'targets':T[tr]}})['fit']; return a,p,terms,q,Y,np.column_stack((fit[:,0],fit[:,1:]@q))

def check(tr,va,debug=False):
 a,p,terms,q,Y,kern=setup(tr,va); errs={}
 def score(name,pred):
  err=pred-Y[va]; errs[name]=[np.mean(abs(err[:,0])),np.mean(abs(err[:,1:]@q.T))]
 score('kernel',kern)
 for nterm in (36,len(terms)):
  yy=[]
  for k in (0,1,2,3,4,6,8,12):
   pred=[]; select=[]
   for sl in (slice(0,1),slice(1,3)):
    cs=path(a[tr],p[tr,:nterm],Y[tr,sl],k); f=np.column_stack((a,p[:,cs])); pred.append(f[va]@lst(f[tr],Y[tr,sl])); select.append([terms[j] for j in cs]);
   score((nterm,k),np.column_stack(pred))
   if debug and k in (2,4,8): print(nterm,k,select,np.round(errs[nterm,k],5))
 f=np.column_stack((a,p[:,:36])); score('fullquad',f[va]@lst(f[tr],Y[tr]));
 return errs

if __name__=='__main__':
 tr=np.arange(52); va=np.arange(52,n); e=check(tr,va,True); print('official',e)
 from sklearn.model_selection import RepeatedKFold
 splits=list(RepeatedKFold(n_splits=4,n_repeats=5,random_state=11).split(T)); es=[check(tr,va) for tr,va in splits]; print('repeated mean / worst')
 for k in e:
  out=np.array([r[k] for r in es]); print(k, np.round(out.mean(0),5),np.round(out.max(0),5))
