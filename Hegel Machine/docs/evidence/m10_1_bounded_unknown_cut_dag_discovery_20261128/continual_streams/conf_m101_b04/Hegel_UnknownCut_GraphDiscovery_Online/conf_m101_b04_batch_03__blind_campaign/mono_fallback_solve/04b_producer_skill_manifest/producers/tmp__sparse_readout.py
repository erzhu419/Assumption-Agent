import importlib.util, pickle
from itertools import combinations
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
spec=importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl','rb') as f: s=pickle.load(f)[0]['study']
x,y,Y=(s['train'][k] for k in ('x','y','traj')); o,g=m._calibration(s['panel_a']); w=(x-o)/g
u,v=(s['panel_b'][k] for k in ('u','v'))
ops={'baseline':m._operator(s['panel_b'])(w)}
for name,fun,start in [('sin',np.sin,[-.4,2.8]),('erf',erf,[-.66,1.45]),('rational',lambda q:q/(1+q*q),[-1.2,.89]),('exp',lambda q:q*np.exp(-q*q),[-1.1,1.0])]:
 def design(a,h):return m._affine(a+h[0]*fun(h[1]*a))
 def res(h):
  a=design(u,h); return (a@np.linalg.lstsq(a,v,rcond=None)[0]-v).ravel()
 h=least_squares(res,start,bounds=([-5,.15],[5,5])).x
 ops[name]=design(w,h)@np.linalg.lstsq(design(u,h),v,rcond=None)[0]
ops['blend']=(ops['sin']+ops['erf']+ops['baseline'])/3
all_i=np.arange(len(x)); folds=[('public',all_i[52:])]+[(f'random{k}',np.random.default_rng(k).permutation(len(x))[-20:]) for k in range(16)]+[(f'axis{k}{sgn:+}',np.argsort(sgn*x[:,k])[-16:]) for k in range(4) for sgn in (1,-1)]+[('radial',np.argsort(np.linalg.norm(x-x.mean(0),axis=1))[-16:])]
pairs=list(combinations(range(4),2))
times=s['eval_times']; B=lambda rates:-np.expm1(-times[:,None]*rates)
mode_data=[]
for _,j in folds:
 i=np.setdiff1d(all_i,j)
 def res(r):
  b=B(r);return (b@np.linalg.lstsq(b,Y[i].T,rcond=None)[0]-Y[i].T).ravel()
 b=B(least_squares(res,[1.8,.4],bounds=([.7,.03],[5.,.68])).x); mode_data.append(np.linalg.qr(b)[0])
for name,z in ops.items():
 a=m._affine(z); pr=np.column_stack([z[:,i]*z[:,j] for i,j in pairs]); errors=np.empty((len(folds),5)); selections=[]; te=[]
 for row,((_,j),b) in enumerate(zip(folds,mode_data)):
  i=np.setdiff1d(all_i,j); best=None
  for cols in combinations(range(len(pairs)),2):
   d=np.c_[a,pr[:,cols]]; c=m._ridge(d[i],y[i],.001); mse=np.mean((d[i]@c-y[i])**2)
   if best is None or mse<best[0]: best=(mse,cols,d,c)
  _,cols,d,c=best; selections.append(tuple(pairs[k] for k in cols));
  for q,D,p in [(0,d,.001),(1,np.c_[a,pr],.001),(2,m._quadratic(z),.01),(3,np.c_[d,w],.008),(4,np.c_[d,d[:,5:]**2],.015)]:
   errors[row,q]=np.mean(abs(D[j]@m._ridge(D[i],y[i],p)-y[j]))
  e=abs(a[j]@m._ridge(a[i],Y[i]@b,.001)@b.T-Y[j]);te.append([e.mean(),e.mean(0).max()])
 print(name,'pairs and counts',[(c,selections.count(c)) for c in sorted(set(selections))], 'trajectory public/worst',np.round([te[0],np.max(te,axis=0)],5).tolist())
 for q,method in enumerate(['two pairs','all pairs','full quadratic','two pairs + input','two pairs + squared interactions']):
  e=errors[:,q]; k=e.argmax();print(method, 'public/random-mean/worst',np.round([e[0],e[1:17].mean(),e[k]],5),folds[k][0])
