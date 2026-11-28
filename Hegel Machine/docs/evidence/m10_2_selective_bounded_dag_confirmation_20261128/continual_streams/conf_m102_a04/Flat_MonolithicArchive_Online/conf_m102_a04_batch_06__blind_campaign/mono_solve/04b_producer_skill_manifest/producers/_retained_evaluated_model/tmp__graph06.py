from probe06 import *
import runpy
from itertools import combinations
from sklearn.linear_model import Ridge
m=runpy.run_path('/tmp/hegel_continual_archive/conf_m102_a04_batch_05_solver.py')
u,v=[s['panel_a'][k] for k in ['u','v']]; keep=np.linalg.norm(v-u,axis=1)<5; u=u[keep];v=v[keep]
for shift in range(1,4):
 j=(np.arange(4)+shift)%4
 f=np.stack((np.ones_like(u),u,u[:,j],u*u[:,j],u**3),axis=-1)
 w=np.stack([ls(f[:,i],v[:,i]) for i in range(4)]);print('A shift',shift,'RMS',np.sqrt(np.mean((np.sum(f*w,axis=-1)-v)**2)),np.round(w,5))
x=s['train']['x']; z=m['_calibrate'](s['panel_a'],x)
b=s['panel_b']; W=ls(bias(b['u']),b['v']); b=bias(z)@W
c=s['panel_c']; g,h=m['_operators'](c,z); gb,hb=m['_operators'](c,b); gx,hx=m['_operators'](c,x)
D=dict(x=x,z=z,b=b,g=g,h=h,G=gb,H=hb,P=gx,Q=hx)
Y=np.c_[s['train']['y'],s['train']['traj']];
def score(f):
 f=bias(f); mu=f[:52].mean(0); sd=np.maximum(f[:52].std(0),1e-9); f[:,1:]=(f[:,1:]-mu[1:])/sd[1:]; out=[]
 for alpha in [0,0.001,0.03,0.3,2]:
  pred=f@ls(f[:52],Y[:52]) if not alpha else Ridge(alpha=alpha).fit(f[:52,1:],Y[:52]).predict(f[:,1:]); e=np.abs(pred-Y); out.append((e[52:,0].mean(),e[52:,1:].mean(),alpha,e[:52,0].mean(),e[:52,1:].mean()))
 return out
R=[]
def add(name,f):
 for a,b,alpha,c,d in score(f):R.append((a,b,name,alpha,c,d))
def cross(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
for n,a in D.items():
 add(n,a); add(n+' poly2',poly(a,2)[:,1:]);add(n+' poly3',poly(a,3)[:,1:]);
for i,j in combinations(D,2):
 a,b=D[i],D[j]; k=i+j
 add(k,np.c_[a,b]); add(k+' dot',np.c_[a,b,np.sum(a*b,axis=1)]); add(k+' diag',np.c_[a,b,a*b]); add(k+' cross',np.c_[a,b,cross(a,b)]); add(k+' poly2',poly(np.c_[a,b],2)[:,1:]);
for keys in combinations(D,3):
 a,b,c=[D[k] for k in keys]; add(''.join(keys),np.c_[a,b,c]); add(''.join(keys)+' diag',np.c_[a,b,c,a*b,b*c,a*c,a*b*c]); add(''.join(keys)+' cross',np.c_[a,b,c,cross(a,b),cross(a,c),cross(b,c)])
for i in [0,1]:
 print('best',i)
 for r in sorted(R,key=lambda r:r[i])[:19]:print(r)
print('ranges',[(k,np.abs(v).max()) for k,v in D.items()]); print('A recovered',np.linalg.norm(z-x,axis=1).max())
