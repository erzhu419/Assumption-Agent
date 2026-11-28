import numpy as np,pickle,itertools
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']

def fit(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def affine(z):return np.c_[np.ones(len(z)),z]
def lib(z):
 d={}
 def add(n,a):d[n]=a
 for i,x in enumerate(z.T):
  for p in range(2,6):add(f'x{i}^{p}',x**p)
  for k in (.15,.22,.2962,.4,.55,.75,1.,1.25,1.5,2.):
   for n,f in [('sin',np.sin(k*x)),('cos',np.cos(k*x)),('tanh',np.tanh(k*x)),('exp',np.expm1(-k*x*x)),('rat',x/(1+k*x*x)),('arctan',np.arctan(k*x)),('log',np.log1p(k*x*x))]:add(f'{n}({k}*x{i})',f)
 for i,j in itertools.combinations(range(4),2):
  x,y=z[:,i],z[:,j]
  for p,q in [(1,1),(1,2),(2,1),(2,2),(3,1),(1,3)]:add(f'x{i}^{p}x{j}^{q}',x**p*y**q)
  for p in (.25,.5,.75,1.,1.5,2.):
   add(f'rat{p} x{i}x{j}',x*y/(1+p*(x*x+y*y)))
   for sg in (-1,1):
    u=x+sg*y
    for n,f in [('sin',np.sin(p*u)),('cos',np.cos(p*u)),('tanh',np.tanh(p*u)),('exp',np.expm1(-p*u*u)),('rat',u/(1+p*u*u))]:add(f'{n} {p} {i},{j},{sg}',f)
 for ids in itertools.combinations(range(4),3):add(str(ids),z[:,ids].prod(1))
 return d

def greedy(z,y,steps=7):
 A=affine(z); D=lib(z); names=list(D); F=np.column_stack(list(D.values())); picked=[]
 for _ in range(steps):
  r=y-A@fit(A,y); G=F-A@fit(A,F); g=(G.T@r)/np.maximum(np.linalg.norm(G,axis=0),1e-10)[:,None]; j=np.argmax(np.linalg.norm(g,axis=1)); A=np.c_[A,F[:,j]]; picked.append(names[j]); print(names[j],round(np.sqrt(np.mean((y-A@fit(A,y))**2)),6),end='; ')
 print()
if __name__=='__main__':
 p=s['panel_c']; u=p['u']; print('panel C sparse grammar:');
 for k in ['v1','v2']:
  y=p[k]; print(k,'all');greedy(u,y)
  for j in range(4): print(k,j,end=' ');greedy(u,y[:,j:j+1],5)
  print('columnwise nonlinear fits');
  for name,fn in [('exp',lambda z,k:np.expm1(-k*z*z)),('cos',lambda z,k:np.cos(k*z)),('sin',lambda z,k:np.sin(k*z)),('tanh',lambda z,k:np.tanh(k*z)),('poly',lambda z,k:z**2),('rat',lambda z,k:z/(1+k*z*z))]:
   def err(k):
    A=np.c_[affine(u),fn(u,k)]; return np.sqrt(np.mean((A@fit(A,y)-y)**2))
   opt=minimize_scalar(err,bounds=(.02,3),method='bounded'); print(name,round(opt.x,5),round(opt.fun,7))
 print('Input sizes',[(k,np.min(s[k]['u'],axis=0),np.max(s[k]['u'],axis=0)) for k in ['panel_a','panel_b','panel_c']]); print('train x sizes',s['train']['x'].min(0),s['train']['x'].max(0))
