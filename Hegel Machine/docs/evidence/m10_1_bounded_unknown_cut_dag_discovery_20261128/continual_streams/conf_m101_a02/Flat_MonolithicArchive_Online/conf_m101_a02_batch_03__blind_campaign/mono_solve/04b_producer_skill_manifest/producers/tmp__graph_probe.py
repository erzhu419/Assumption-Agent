import pickle,itertools,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
fit=lambda x,y: np.linalg.lstsq(x,y,rcond=None)[0]
def aff(x): return np.c_[np.ones(len(x)),x]
def pol(x,d):
 return np.column_stack([np.prod(x[:,p],axis=1) for k in range(d+1) for p in itertools.combinations_with_replacement(range(4),k)])
def calibration(p):
 u,v=p['u'],p['v']; a=aff(u); keep=np.ones(len(u),bool)
 for _ in range(3):
  r=np.linalg.norm(a@fit(a[keep],v[keep])-v,axis=1); keep=r<3*np.median(r)
 j=np.roll(u,-1,axis=1); D=np.stack([np.ones_like(u),u,j,u*j,u**3],2)
 c=np.column_stack([fit(D[keep,i],v[keep,i]) for i in range(4)])
 print('calibration',c,'inlier rms',np.std(np.sum(D[keep]*c.T,axis=2)-v[keep])); return c
def invert(x,c):
 u=(x-c[0])/c[1]; i=np.arange(4); j=(i+1)%4
 for _ in range(30):
  uj=u[:,j]; r=c[0]+c[1]*u+c[2]*uj+c[3]*u*uj+c[4]*u**3-x; J=np.zeros((len(x),4,4)); J[:,i,i]=c[1]+c[3]*uj+3*c[4]*u**2; J[:,i,j]=c[2]+c[3]*u
  u-=np.linalg.solve(J,r[...,None])[...,0]
 print('inverse max residual',np.max(np.abs(r))); return u
x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; m=52
c=calibration(s['panel_a']); u=invert(x,c); p=s['panel_b']; b=fit(aff(p['u']),p['v']); z=aff(u)@b

def library(a):
 a0,a1,a2,a3=a.T; pair=a0*a2+a1*a3; h=a0+a1-a2-a3
 return np.c_[aff(a),pair,np.sin(.2*h),np.tanh(h),pair/(1+h*h)]
def report(name,D):
 c=fit(D[:m],np.c_[y,T][:m]); pred=D@c; E=np.abs(pred-np.c_[y,T]); print(name,D.shape[1],'fit',np.round([E[:m,0].mean(),E[:m,1:].mean()],5),'52/20',np.round([E[m:,0].mean(),E[m:,1:].mean()],5))
for n,a in [('raw',x),('cal',u),('mixer',z)]:
 for d in [1,2,3]: report(n+str(d),pol(a,d))
 report(n+'archive readout',library(a))
p=s['panel_c']; a=p['u']
for k in ['v1','v2']:
 v=p[k]; print('C',k)
 for tag,D in [('diag quad',np.c_[aff(a),a**2]),('diag cubic',np.c_[aff(a),a**2,a**3]),('odd cubic',np.c_[aff(a),a**3]),('harmonic',np.c_[aff(a),np.sin(1.7*a),np.cos(1.7*a)]),('tanh',np.c_[aff(a),np.tanh(a)])]:
  cf=fit(D,v); e=D@cf-v; h=np.sum(D*np.linalg.pinv(D).T,axis=1); print(tag,np.std(e),np.std(e/(1-h[:,None])))
 for d in [1,2]:
  cf=fit(pol(a,d),v)
  for tag,q in [('x',x),('u',u),('z',z)]:
   w=pol(q,d)@cf; report('C '+k+str(d)+'('+tag+')',np.c_[aff(z),w]); report('C nonlinear '+k+str(d)+'('+tag+')',library(w))
