import pickle, numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
frozen={}; exec(open('/holdout_task/_frozen_nodes/node_01.py').read(),frozen)
def affine(x):return np.c_[np.ones(len(x)),x]
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
u,v=(s['panel_a'][k] for k in ('u','v'))
a=affine(u); m=np.ones(len(u),bool)
for _ in range(6):
 r=np.linalg.norm(a@ls(a[m],v[m])-v,axis=1);m=np.argsort(r)[:len(u)-3]
coef=[]
for j in range(4):
 k=(j+1)%4; D=np.c_[np.ones(len(u)),u[:,j],u[:,k],u[:,j]*u[:,k],u[:,j]**3]; c=ls(D[m],v[m,j]);coef.append(c);print('a channel',j,c.round(6),'rms',np.sqrt(np.mean((D[m]@c-v[m,j])**2)))
c=np.array(coef); b=ls(affine(s['panel_b']['u']),s['panel_b']['v'])
def A(x):
 n=np.roll(x,-1,axis=-1);return c[:,0]+c[:,1]*x+c[:,2]*n+c[:,3]*x*n+c[:,4]*x**3
def Ai(x):
 out=[]
 for row in x:
  q=least_squares(lambda z:A(z)-row,(row-c[:,0])/c[:,1],gtol=1e-12)
  out.append(q.x)
 return np.array(out)
def B(x):return affine(x)@b
def Bi(x):return np.linalg.solve(b[1:].T,(x-b[0]).T).T
x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']
trans={'raw':lambda x:x,'A':A,'Ai':Ai,'B':B,'Bi':Bi,'BAi':lambda x:B(Ai(x)),'BiAi':lambda x:Bi(Ai(x)),'AiBi':lambda x:Ai(Bi(x)),'AiB':lambda x:Ai(B(x)),'BA':lambda x:B(A(x)),'AB':lambda x:A(B(x))}
from scipy.optimize import minimize_scalar
p=s['panel_c']; u=p['u']; C={}
for key,fun in [('v1',np.tanh),('v2',np.cos)]:
 def basis(z,a): return np.c_[affine(z),fun(a*z)]
 def obj(a):
  D=basis(u,a); e=p[key]-D@ls(D,p[key]); return np.sum(e*e)
 a=minimize_scalar(obj,bounds=(.6,3.5),method='bounded').x
 w=ls(basis(u,a),p[key]); print(key,'frequency/RMS',a,(obj(a)/96)**.5,'coef',w.round(4)); C[key]=(lambda z,a=a,w=w,basis=basis,f=fun:np.c_[affine(z),f(a*z)]@w)
from itertools import combinations, combinations_with_replacement
from scipy.optimize import minimize_scalar
z=Ai(x); q=Bi(z); Y=np.c_[t,y]; tr=np.arange(52); te=np.arange(52,72)
results=[]
def check(name,D):
 D=affine(D); w=ls(D[tr],Y[tr]); E=D@w-Y
 results.append((np.mean(abs(E[te,:-1])),np.mean(abs(E[tr,:-1])),np.mean(abs(E[te,-1])),name));return w
for name,g in [('Ai',z),('BiAi',q),('BAi',B(z)),('raw',x)]:
 i,j=np.triu_indices(4,1); check(name,g)
 for deg in (2,3):
  D=np.column_stack([np.prod(g[:,term],axis=1) for k in range(1,deg+1) for term in combinations_with_replacement(range(4),k)]); w=check(name+str(deg),D); print('polynomial',name,deg,results[-1]);
  if name=='Ai': print('Ai trajectory polynomial terms',sorted(zip(np.linalg.norm(w[1:,:-1],axis=1).round(4),[term for k in range(1,deg+1) for term in combinations_with_replacement(range(4),k)]),reverse=True)[:15]);
 for f in (np.sin,np.cos,np.tanh,np.sinh,np.exp):
  for k in (.5,1,1.3,1.7,2,2.4,3,4,5):
   for coeff in (.1,.2,.4,1.):
    check(f'{name} + {coeff} {f.__name__} {k}',g+coeff*f(k*g));check(f'{name} - {coeff} {f.__name__} {k}',g-coeff*f(k*g));
   check(f'{name} full {f.__name__} {k}',np.c_[g,f(k*g),g[:,i]*g[:,j]])
 for M in (z,q,B(z),C['v1'](z),C['v2'](z),C['v1'](q),C['v2'](q)):
  for k in (.5,1.,1.3,1.7,2.4,3.4):
   for f in (np.tanh,np.sin,np.cos): check(f'{name} {f.__name__} {k} {np.linalg.norm(M):.4f}',np.c_[g,f(k*M)])
print('best',*sorted(results)[:20],sep='\n');print('best train',*sorted(results,key=lambda r:r[1])[:12],sep='\n')
print('time',s['eval_times']);print('bounds train, test',np.min(z,0),np.max(z,0),np.min(Ai(s['test']['x']),0),np.max(Ai(s['test']['x']),0));
