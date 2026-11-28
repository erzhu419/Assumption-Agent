import numpy as np,pickle,runpy
from scipy.optimize import minimize_scalar
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; T=tr['traj']; t=s['eval_times']; w=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':s})['w'][:len(y)]
def aff(x):return np.c_[np.ones(len(x)),x]
def fit(x,z):return np.linalg.lstsq(x,z,rcond=None)[0]
def poly(x,d):return np.column_stack([np.prod(x[:,k],axis=1) for j in range(d+1) for k in combinations_with_replacement(range(x.shape[1]),j)])
def errs(a):
 z=np.c_[y,T]; r=a@fit(a,z)-z; p=a[52:]@fit(a[:52],z[:52])-z[52:]; return np.round([abs(r[:,0]).mean(),abs(r[:,1:]).mean(),abs(p[:,0]).mean(),abs(p[:,1:]).mean()],5)
from scipy.optimize import least_squares
from itertools import combinations, combinations_with_replacement
from scipy.optimize import least_squares
np.set_printoptions(precision=4,suppress=True)
b=s['panel_b'];c=s['panel_c'];bx,bv=b['u'],b['v'];cx,cv,dv=c['u'],c['v1'],c['v2'];B=fit(aff(bx),bv);z=np.linalg.solve(B[1:].T,(w-B[0]).T).T
print('panel sizes',[(k,{j:np.shape(a) for j,a in s[k].items()}) for k in ('panel_a','panel_b','panel_c')]);print('times',t)
pairs=list(combinations(range(4),2)); triples=list(combinations_with_replacement(range(4),3)); triples=[k for k in triples if len(set(k))>1]
def prod(x,ks):return np.column_stack([np.prod(x[:,k],axis=1) for k in ks])
def cvstats(a,v):
 h=np.sum(np.linalg.svd(a,full_matrices=False)[0]**2,axis=1); e=v-a@fit(a,v);return np.std(e),np.sqrt(np.mean((e/(1-h[:,None]))**2))
models={}
def add(name,f):
 a=f(bx);q=fit(a,bv);v=f(z)@q;models[name]=v;print(name,'panel',np.round(cvstats(a,bv),5))
 return q
f0=lambda x:np.c_[aff(x),x**3]
for ks in [[],pairs,triples[:0],list(combinations_with_replacement(range(4),2))]:
 if ks:add('base+'+str(len(ks)),lambda x:np.c_[f0(x),prod(x,ks)])
extra={str(k):lambda x,k=k:np.prod(x[:,k],axis=1) for k in list(combinations_with_replacement(range(4),2))+triples}
extra.update({f'p{p}i{i}':lambda x,p=p,i=i:x[:,i]**p for p in (4,5,7) for i in range(4)})
chosen=[]
for k in range(5):
 def f(x,keys):return np.column_stack([f0(x)]+[extra[j](x) for j in keys]);
 key=min((q for q in extra if q not in chosen),key=lambda q:cvstats(f(bx,chosen+[q]),bv)[1]);chosen.append(key)
 add('step'+str(k),lambda x:f(x,chosen));print('selected',chosen)
for label,fun,x0 in [('cubic',lambda x,p:x**3+x*p,np.full(4,.22)),('cubicshift',lambda x,p:(x+p[4:])**3+x*p[:4],np.r_[np.full(4,.22),np.zeros(4)]),('sinh',lambda x,p:np.sinh(x*p),np.full(4,2.)),('mix',lambda x,p:x**3+x*p[0],np.array([.22])),('powers',lambda x,p:np.sign(x)*abs(x)**p[0]+x*p[1],np.array([3,.22]))]:
 for keys in [[],pairs]:
  def ff(x,p):return np.column_stack([aff(fun(x,p))]+([prod(x,keys)] if keys else []))
  def r(p):
   a=ff(bx,p);return (a@fit(a,bv)-bv).ravel()
  q=least_squares(r,x0,max_nfev=100);print(label,keys,q.x);add(label+str(len(keys)),lambda x:ff(x,q.x))
models['base']=f0(z)@fit(f0(bx),bv)
fn={'atan':np.arctan,'asinh':np.arcsinh,'tanh':np.tanh,'inv3':lambda u:np.sinh(np.arcsinh(u)/3)}
c2=lambda u:poly(u,2)@fit(poly(cx,2),dv)
print('C2 coefficients',fit(poly(cx,2),dv))
for name,v in models.items():
 for k,f in fn.items():
  g=minimize_scalar(lambda g:np.mean((aff(f(cx*g))@fit(aff(f(cx*g)),cv)-cv)**2),bounds=(.1,8),method='bounded').x
  d=aff(f(v*g)); h=d@fit(aff(f(cx*g)),cv); e=c2(v)
  E=[errs(a) for a in (d,np.c_[d,e],np.c_[d,v],np.c_[d,e,v],np.c_[d,prod(h,pairs)],np.c_[d,e,prod(h,pairs)],np.c_[aff(v),e])]; best=min(range(len(E)),key=lambda i:max(E[i][2:]));print(name,k,best,E[best],E[0])
