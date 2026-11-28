import pickle
import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from scipy.optimize import least_squares, minimize_scalar
from itertools import combinations_with_replacement
from importlib.util import spec_from_file_location,module_from_spec
spec=spec_from_file_location('m','/holdout_task/batch_solver.py');m=module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; t=s['eval_times']; n=len(x); ids=np.arange(n)
def aff(q):return np.c_[np.ones(len(q)),q]
def fit(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
p=s['panel_a'];a=aff(p['u']); j=np.arange(len(a))
for _ in range(4):
 w=fit(a[j],p['v'][j]); j=np.argsort(np.linalg.norm(a@w-p['v'],axis=1))[:-3]
a=(x-w[0])/np.diag(w[1:]);p=s['panel_b'];w=fit(aff(p['u']),p['v']);b=aff(a)@w;br=aff(x)@w
p=s['panel_c'];u=p['u']
def c1(q,k=.834):return np.c_[aff(q),q/(1+k*abs(q))]
v1=fit(c1(u),p['v1']);
def c2(q,k=.77):return np.c_[aff(q),np.arcsinh(k*q)**2]
v2=fit(c2(u),p['v2'])
coords={'raw':x,'Ainv':a,'B':b,'Braw':br,'Craw':c1(x)@v1,'CA':c1(a)@v1,'CBA':c1(b)@v1,'CB':c1(br)@v1,'Draw':c2(x)@v2,'DA':c2(a)@v2,'DBA':c2(b)@v2,'DB':c2(br)@v2}
print('panel C2 trial fit',np.sqrt(np.mean((p['v2']-c2(u)@v2)**2)))
f=m._quadratic(x); poly=f@f.T; d=cdist(x,x,'sqeuclidean');v=np.sqrt(5*d)/3; kr=np.exp(-d/8); km=(1+v+v*v/3)*np.exp(-v)
sy=poly+kr; sz=poly+.1*km
R=np.sum(x*x,1); splits=list(np.array_split(ids,6))+[np.argsort(R)[-12:],np.argsort(R)[:12],ids[56:]]
def pred(K,i,j,Y,noise):return K[np.ix_(j,i)]@solve(K[np.ix_(i,i)]+noise*np.eye(len(i)),Y,assume_a='pos')
Y=np.column_stack((y,z));
def projected(i):
 u=m._temporal_basis(t,z[i]);return (u@fit(u,z[i].T)).T
Z=[projected(np.setdiff1d(ids,j)) for j in splits]
rows=[]
def score(name,K1,K2):
 r=[]
 for j,zz in zip(splits,Z):
  i=np.setdiff1d(ids,j);h=abs(pred(K1,i,j,y[i],.0001)-y[j]); e=abs(pred(K2,i,j,zz,.00003)-z[j]);r.append([h.mean(),e.mean(),e[:,-1].mean(),e.mean(0).max()])
 r=np.array(r);rows.append((name,r[:6].mean(0),r.max(0),r[-1]))
score('reference',sy,sz)
for name,q in coords.items():
 d=cdist(q,q,'sqeuclidean'); sd=np.median(d[d>0]); dc=(q[:,None,:]-q[None,:,:])**2; dc/=np.maximum(q.var(0),1e-8)
 qf=m._quadratic(q); qpoly=qf@qf.T; qpoly/=np.trace(qpoly)/n
 for kn,h in [('joint',np.exp(-d/sd)),('add',np.exp(-dc/4).mean(2)),('tight',np.exp(-dc/1.5).mean(2)),('quadratic',qpoly)]:
  for amp in (.02,.15,1.):score((name,kn,amp),sy+amp*h,sz+amp*h)
for i in (0,1,3):
 print('best',i)
 for name,avg,worst,tail in sorted(rows,key=lambda w:w[1][i]+.7*w[2][i])[:6]:print(name,avg.round(5),worst.round(5),tail.round(5))
print('reference',rows[0][1:]);print('train/test coordinate support',np.max(np.linalg.norm(x,axis=1)),np.max(np.linalg.norm(s['test']['x'],axis=1)))
