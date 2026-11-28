exec(open('/tmp/probe05.py').read().split('for k,w in coords.items():')[0])
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
for target in ['v1','v2']:
 u=c['u']; v=c[target]
 for base in (False,True):
  def design(w,a):
   f=w/(1+a*abs(w)) if target=='v1' else np.log(np.cosh(a*w)); return np.c_[A(w) if base else np.ones(len(w)),f]
  def loss(a):
   F=design(u,a);return np.mean((v-F@np.linalg.lstsq(F,v,rcond=None)[0])**2)
  opt=minimize_scalar(loss,bounds=(.25,1.8),method='bounded');print('panel',target,base,opt.x,np.sqrt(opt.fun));
  if base:
   cb=np.linalg.lstsq(design(u,opt.x),v,rcond=None)[0];
   for k in ['z','h','Bz']:
    coords[k+target]=design(coords[k],opt.x)@cb

def Fscore(F,lam=1e-6):
 F=F/np.maximum(np.std(F,axis=0),.03);return evaluate(F@F.T+lam*np.eye(len(F)))
for k in ['x','z','h','Bz','zv1','zv2']:
 w=coords[k];
 features={'lin':A(w),'quad':np.c_[A(w),*[w[:,i]*w[:,j] for i,j in combinations_with_replacement(range(4),2)]], 'addpoly':np.c_[A(w),w**2,w**3,w**4,w**5], 'addsmooth':np.c_[A(w),np.sin(w),np.cos(w),np.tanh(1.25*w),np.log(np.cosh(1.25*w)),w/(1+.22*w*w)],'addpanel':np.c_[A(w),w/(1+.84*abs(w)),np.log(np.cosh(1.1*w))]}
 for nm,F in features.items():print(k,nm,np.round(Fscore(F,1e-5),4))
for a,b in [('z','zv1'),('z','zv2'),('h','hv1'),('h','hv2'),('Bz','Bzv1'),('Bz','Bzv2'),('zv1','zv2'),('hv1','hv2'),('Bzv1','Bzv2')]:
 w,v=coords[a],coords[b]
 for order in [1,2]:
  F=np.c_[A(w),v] if order==1 else np.c_[A(w),v,*[a[:,i]*a[:,j] for a in (w,v) for i,j in combinations_with_replacement(range(4),2)]]
  print(a,b,order,np.round(Fscore(F,1e-4),4))
tt=s['eval_times']; U,S,V=np.linalg.svd(t,full_matrices=False)
def Tfit(k):
 F=1-np.exp(-k*tt); amp=t@F/(F@F);return np.std(t-amp[:,None]*F),amp
op=minimize_scalar(lambda a:Tfit(a)[0],bounds=(.1,3),method='bounded');print('time',op.x,op.fun,'fixed',[(z,Tfit(z)[0]) for z in (.22,.75,1.25)]); print('time2',np.round(V[1],4),'rate-shape corr',np.corrcoef(V[1],tt*np.exp(-op.x*tt))[0,1]);
