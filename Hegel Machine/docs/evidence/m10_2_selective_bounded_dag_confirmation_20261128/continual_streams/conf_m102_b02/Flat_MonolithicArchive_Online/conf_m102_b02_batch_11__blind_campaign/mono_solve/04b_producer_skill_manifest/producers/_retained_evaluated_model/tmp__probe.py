import numpy as np,pickle,glob,importlib.util,copy
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; Y=np.c_[tr['y'],tr['traj']]; n=52
I=lambda a:np.c_[np.ones(len(a)),a]
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
def poly(a,d):
 return np.column_stack([np.prod(a[:,j],axis=1) for k in range(d+1) for j in combinations_with_replacement(range(a.shape[1]),k)])
def score(f):
 p=f[n:]@ls(f[:n],Y[:n]); e=abs(p-Y[n:]);return np.round([e[:,0].mean(),e[:,1:].mean()],6)
p=s['panel_a']; u=p['u']; v=p['v']; a=I(u); keep=np.ones(len(u),bool)
for _ in range(8):
 c=ls(a[keep],v[keep]); r=np.linalg.norm(a@c-v,axis=1); keep=r<max(.003,4*np.median(r))
print('A',keep.sum(),c,'err',np.sqrt(np.mean((a[keep]@c-v[keep])**2)))
for d in [2,3]:
 a2=poly(u,d); print('A poly',d,np.sqrt(np.mean((a2[keep]@ls(a2[keep],v[keep])-v[keep])**2)))
z=np.linalg.solve(c[1:].T,(x-c[0]).T).T
p=s['panel_b']; b=ls(I(p['u']),p['v']); w=I(z)@b
p=s['panel_c']; u=p['u']; pair=np.triu_indices(4,1); F={'affine':I,'pairs':lambda a:I(a[:,pair[0]]*a[:,pair[1]]),'square':lambda a:I(a*a),'poly2':lambda a:poly(a,2),'cube':lambda a:I(np.c_[a,a**3]),'tanh':lambda a:I(np.tanh(2*a))}; C=[]
for k,v in p.items():
 if k=='u':continue
 for name,f in F.items():
  a=f(u); h=a@ls(a,v)-v; lev=np.sum(a*np.linalg.pinv(a).T,axis=1); print('C',k,name,np.sqrt(np.mean(h**2)),np.sqrt(np.mean((h/(1-lev[:,None]))**2)))
  for label,base in [('z',z),('w',w)]:
   q=f(base)@ls(a,v); print('    ',label,score(I(np.c_[z,q])),score(I(np.c_[z,q,(z[:,:,None]*q[:,None,:]).reshape(len(x),-1)])))
for label,base in [('x',x),('z',z),('w',w)]:
 for d in [1,2,3]:print(label,d,score(poly(base,d)))
for fn in sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py')):
 try:
  spec=importlib.util.spec_from_file_location('candidate',fn); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  st=copy.deepcopy(s);st['train']={k:v[:n] if k!='summary' else v for k,v in tr.items()}; st['test']['x']=x[n:]; out=m.solve_batch(st); e=abs(np.c_[out['y_hat'],out['state_pred']]-Y[n:]);print(fn.split('/')[-1], e[:,0].mean(),e[:,1:].mean())
 except Exception as e:print(type(e),e)
