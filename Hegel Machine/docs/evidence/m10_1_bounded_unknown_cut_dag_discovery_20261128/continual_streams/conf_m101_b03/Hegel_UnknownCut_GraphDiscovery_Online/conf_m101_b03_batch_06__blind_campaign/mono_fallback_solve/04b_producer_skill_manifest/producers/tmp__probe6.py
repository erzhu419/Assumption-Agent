import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from graph_probe import *
from itertools import product
from scipy.optimize import minimize_scalar
from sklearn.model_selection import KFold
from importlib.util import spec_from_file_location,module_from_spec
import glob,copy
for path in sorted(glob.glob('/tmp/hegel_continual_archive/*_solver.py')):
 spec=spec_from_file_location('prior',path); m=module_from_spec(spec); spec.loader.exec_module(m)
 ss=copy.deepcopy(s); ss['test']['x']=x[n:]; ss['train'].update(x=x[:n],y=y[:n],traj=T[:n]); o=m.solve_batch(ss)
 print('archive',path.rsplit('/',1)[-1],np.mean(abs(o['y_hat']-y[n:])),np.mean(abs(o['state_pred']-T[n:])))
def pair4(z): return np.c_[z[:,0]*z[:,1],z[:,0]*z[:,3],z[:,1]*z[:,2],z[:,2]*z[:,3]]
wc=np.linalg.lstsq(aff(pair4(p['u'])),p['v1'],rcond=None)[0]
def obj(k):
 f=aff(np.cos(k*p['u'])); return np.linalg.norm(f@np.linalg.lstsq(f,p['v2'],rcond=None)[0]-p['v2'])
k=minimize_scalar(obj,bounds=(2.2,2.6),method='bounded').x
wd=np.linalg.lstsq(aff(np.cos(k*p['u'])),p['v2'],rcond=None)[0]
def outer(q,v):return (q[:,:,None]*v[:,None,:]).reshape(len(q),-1)
def sym(z,d):return np.column_stack([(z[:,i]*z[:,j]) for i,j in combinations_with_replacement(range(d),2)])
cv=list(KFold(4,shuffle=True,random_state=6).split(x[:n])); targets=np.c_[y,a,T]
for name,z in list(maps.items())[:6]:
 q=pair4(z); h=np.cos(k*z); cc=aff(q)@wc; dd=aff(h)@wd; bfeat=aff(z)@b
 g={'z':z,'quad':q,'quad+':np.c_[z*z,z[:,0]*z[:,2],z[:,1]*z[:,3]],'cos':h,'sin':np.sin(k*z),'sumsin':np.sin(k*np.c_[[z[:,i]+sg*z[:,j] for i,j in pairs for sg in (-1,1)]].T),'linC':np.c_[cc,dd],'q*h':outer(q,h),'z*h':outer(z,h),'z*q':outer(z,q),'q*q':sym(q,4),'h*h':sym(h,4),'tanh':np.tanh(2*z),'rat':z/(1+z*z),'mod':np.c_[np.linalg.norm(z,axis=1),np.linalg.norm(cc,axis=1),np.linalg.norm(dd,axis=1)]}
 for j in range(4):
  g[f'z.C{j}']=np.sum(z*np.roll(cc,j,axis=1),axis=1)[:,None]
  g[f'z.D{j}']=np.sum(z*np.roll(dd,j,axis=1),axis=1)[:,None]
  g[f'C.D{j}']=np.sum(cc*np.roll(dd,j,axis=1),axis=1)[:,None]
  g[f'B.C{j}']=np.sum(bfeat*np.roll(cc,j,axis=1),axis=1)[:,None]
  g[f'B.D{j}']=np.sum(bfeat*np.roll(dd,j,axis=1),axis=1)[:,None]
 def matrix(keys): return np.column_stack([np.ones(len(z)),*[g[i] for i in keys]])
 def pred(F,tr,te,v): return F[te]@np.linalg.lstsq(F[tr],v[tr],rcond=1e-8)[0]
 def mae(F,v,sl):
  err=pred(F,*sl,v)-v[sl[1]]
  if v.ndim==2: err=err@basis(r).T
  return np.mean(abs(err))
 def search(v):
  def score(keys):
   F=matrix(keys)
   if F.shape[1]>32:return 1e6
   return np.mean([mae(F,v,sl) for sl in cv])
  beam=[(score(()),())]; seen={()}; records=[]
  for _ in range(6):
   nxt=[]
   for _,ks in beam:
    for j in g:
     if j in ks:continue
     kk=tuple(sorted((*ks,j)))
     if kk in seen:continue
     seen.add(kk); nxt.append((score(kk),kk))
   beam=sorted(nxt)[:10];records.extend(beam)
  for cvmae,keys in sorted(records)[:3]:
   F=matrix(keys); print(round(cvmae,5),round(mae(F,v,(np.arange(n),np.arange(n,len(x)))),5),keys,end='; ')
 for label,v in [('scalar',y),('traj',a)]:
  print(name,label,end=' '); search(v); print()
