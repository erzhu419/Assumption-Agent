exec(open('/tmp/panel_probe.py').read().split('for k,v in dict')[0])
from scipy.optimize import least_squares
from itertools import product
arc=runpy.run_path('/tmp/hegel_continual_archive/conf_m102_b02_batch_01_solver.py')
cv=[(np.arange(52),np.arange(52,72))]+[(np.delete(np.arange(72),e),e) for e in np.array_split(np.random.default_rng(52).permutation(72),4)]
U,S,V=np.linalg.svd(y[:,1:],full_matrices=False); Y=np.c_[y[:,0],U[:,:2]*S[:2]]
def choose(d,regs,name):
 items=[]
 for r in regs:
  scores=[]
  for tr,te in cv:
   X=d[tr]; a=np.linalg.solve(X.T@X+r*np.eye(X.shape[1]),X.T@Y[tr]); E=d[te]@a-Y[te]; scores.append([abs(E[:,0]).mean(),abs(E[:,1:]@V[:2]).mean()]);
  items.append((max(np.max(scores,axis=0)),r,np.round(scores[0],4),np.round(np.max(scores,axis=0),4)))
 print(name,min(items,key=lambda x:x[0]),flush=True)
regs=[1e-7,1e-5,.001,.01,.1,1.]
xs={'x':x,'z':z,'z0':x@b[1:]}
for key in ('v1','v2'):
 try:
  xs[key]=arc['_operator'](c,key,z,'c'); pred=arc['_operator'](c,key,c['u'],'c');print('panel C error',key,np.sqrt(np.mean((pred-c[key])**2)))
 except Exception as e: print(e)
for k,q in xs.items():
 for degree in (2,3,4,5):
  D=np.column_stack([np.ones(len(q))]+[q[:,t].prod(1) for t in full(degree)]); choose(D,regs,k+str(degree))
 for trf in (np.tanh,np.sin,np.arctan,lambda t:t/(1+abs(t)),lambda t:t**2,lambda t:t**3):
  for strength in (.5,1,2):
   q1=trf(strength*q); D=np.column_stack([np.ones(len(q)),q,q1]+[q1[:,t].prod(1) for t in full(3) if len(t)>1]); choose(D,regs,k+str(trf)+str(strength))
for k in ('v1','v2'):
 for q in (z,x):
  w=xs[k]; choose(np.column_stack([np.ones(len(q)),q,w,(q[:,:,None]*w[:,None,:]).reshape(len(q),-1)]),regs,k+'cross')
print('poly successive residuals, rank by order')
for tr in (np.arange(52),np.arange(72)):
 for k in (1,2,3,4):
  D=np.column_stack([np.ones(len(z))]+[z[:,t].prod(1) for t in full(k)]); E=Y-D@np.linalg.lstsq(D[tr],Y[tr],rcond=None)[0]; print(len(tr),k,np.sqrt(np.mean(E[tr]**2,axis=0)),np.linalg.svd(E[tr],compute_uv=False))
