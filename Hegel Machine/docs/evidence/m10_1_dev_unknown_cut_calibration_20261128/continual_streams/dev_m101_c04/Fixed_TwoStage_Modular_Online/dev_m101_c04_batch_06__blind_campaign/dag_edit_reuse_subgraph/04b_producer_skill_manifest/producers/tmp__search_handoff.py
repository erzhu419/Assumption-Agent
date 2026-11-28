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
from itertools import product

def inverse(F,z,guess):
 return np.array([least_squares(lambda v:F(v[None])[0]-r,g,max_nfev=80).x for r,g in zip(z,guess)])
for key in ('v1','v2'):
 f=C[key]; lin=ls(affine(p['u']),p[key]); C[key+'i']=lambda z,f=f,lin=lin:inverse(f,z,np.clip(np.linalg.solve(lin[1:].T,(z-lin[0]).T).T,-2,2))
F={**trans,**C}; tr=np.arange(52);te=np.arange(52,len(x)); scored=[]
def score(n,z):
 D=affine(z); e=D[te]@ls(D[tr],t[tr])-t[te]; err=np.mean(abs(e));scored.append((err,n,z));return err
for n,f in F.items():
 z=f(x); score(n,z)
 for k,g in F.items():
  if k in ('raw',) or n=='raw':continue
  score(n+'>'+k,g(z))
for _,n,z in sorted(scored,key=lambda q:q[0])[:16]: print('trajectory',_,n, 'range',np.max(abs(z)))
for err,n,z in sorted(scored,key=lambda q:q[0])[:12]:
 o=frozen['run_node']({'node_00':{'z_tr':z[tr],'z_te':z[te],'y':y[tr],'traj':t[tr]}});print('both',n,err,np.mean(abs(o['y_hat']-y[te])))
print('sample rows',np.c_[x,Ai(x),Bi(Ai(x)),y,t[:,[0,3,7]]][:8].round(3))
