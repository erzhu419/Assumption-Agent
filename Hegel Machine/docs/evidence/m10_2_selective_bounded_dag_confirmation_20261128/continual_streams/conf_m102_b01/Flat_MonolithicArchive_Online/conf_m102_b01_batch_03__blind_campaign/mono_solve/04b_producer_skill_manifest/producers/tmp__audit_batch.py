import pickle,sys,numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0,'/holdout_task')
import batch_solver as b
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; N=len(tr['y']); ii=np.arange(N)
print('study keys:',sorted(s),'train keys:',sorted(tr),'telemetry shape:',np.shape(s['telemetry_vec']))
p=s['panel_a']; u,v=p['u'],p['v']; mag=np.linalg.norm(v,axis=1); med=np.median(mag); clean=mag<=med+10*np.median(abs(mag-med)); F=b._cal_basis(u); errs=[]
for j in np.flatnonzero(clean):
 k=clean.copy(); k[j]=False
 ca=np.array([b._ls(F[k,h],v[k,h]) for h in range(4)]); af=b._ls(b._aff(u[k]),v[k]); errs.append((np.linalg.norm((F[j]*ca).sum(1)-v[j]),np.linalg.norm(b._aff(u[j:j+1])@af-v[j])))
print('Panel A leave-one-clean-row-out norm error: median and max, cubic vs affine',np.median(errs,0),np.max(errs,0),'excluded rows',np.flatnonzero(~clean))
w=b._coordinates(s,tr['x']); splits=[ii[52:]]+list(np.array_split(np.random.default_rng(1323).permutation(N),4))
for j in range(4):
 ix=np.argsort(tr['x'][:,j]); splits.extend([ix[:16],ix[-16:]])
splits.append(np.argsort(np.linalg.norm(tr['x'],axis=1))[-16:]);
def columns(g,kind,model):
 f=b._design(w,g,kind); h=f[:,1:5]; cols=[]
 for j in range(4):
  for k in range(j+1,4):
   for sg in [-1,1]:
    d=h[:,j]+sg*h[:,k]
    if model=='pairs': cols.append(np.arcsinh(d))
    if model=='rational': cols.append(h[:,j]/(1+.22*d*d))
    if model=='third':
     for r in range(k+1,4): cols.append(h[:,j]*h[:,k]*h[:,r])
 if not cols:return f
 return np.c_[f,np.unique(np.asarray(cols).T,axis=1)]
for model in ['base','pairs','rational','third']:
 results=[]
 for v in splits:
  a=np.setdiff1d(ii,v); sc=[]
  for kind,target in [('scalar',tr['y']),('trajectory',tr['traj'])]:
   def loss(g):
    f=columns(g,kind,model)[a]; return np.mean((f@b._ls(f,target[a])-target[a])**2)
   g=minimize_scalar(loss,bounds=(.3,9) if kind=='scalar' else (.25,4),method='bounded').x; f=columns(g,kind,model); e=abs(f[v]@b._ls(f[a],target[a])-target[v]); sc.extend([e.mean(),np.max(e.mean(0))]);
  results.append(sc)
 R=np.array(results); print(model,'52/20',np.round(R[0],6),'worst',np.round(R.max(0),6),'mean',np.round(R.mean(0),6))
print('test latent coordinate ranges',np.round(w.min(0),3),np.round(w.max(0),3),np.round(b._coordinates(s,s['test']['x']).min(0),3),np.round(b._coordinates(s,s['test']['x']).max(0),3))
