exec(open('/tmp/challenge.py').read().split('R=[]')[0])
from scipy.optimize import minimize_scalar
families={'sin':(np.sin,2,4),'rat':(lambda a:a/(1+a*a),.3,1.5),'tanh':(np.tanh,.65,2.5),'gaussian':(lambda a:a*np.exp(-a*a),.6,2.5),'erf':(__import__('scipy.special',fromlist=['erf']).erf,.6,2.5),'rational4':(lambda a:a/(1+a**4),.4,1.5),'softcube':(lambda a:a**3/(1+a**6)**.5,.6,2.5)}
p=s['panel_b']; U,V=p['u'],p['v']; D=s['panel_c'];
def panel(u,v,f):
 A=np.c_[np.ones(len(u)),u,f(u)]; H=A@np.linalg.pinv(A); E=(v-H@v)/np.maximum(1-np.diag(H),.001)[:,None]; return np.sqrt(np.mean(E**2))
for k,(f,lo,hi) in families.items():
 op=minimize_scalar(lambda a:panel(U,V,lambda u:f(a*u)),bounds=(lo,hi),method='bounded'); fn=lambda x:f(op.x*x); a,e=des(X,fn,(0,1,3,4,5,6)); v=cv(a,e,.1,1); print(k,round(op.x,5),'panel B loo',round(op.fun,5),'panel C',round(panel(D['u'],D['v1'],fn),5),'main split',v[0].round(4),'max',v.max(0).round(4))
# Test whether panel B supplies the spatial modes themselves instead of just the shape.
for k,(f,lo,hi) in families.items():
 op=minimize_scalar(lambda a:panel(U,V,lambda u:f(a*u)),bounds=(lo,hi),method='bounded'); fn=lambda u:np.c_[np.ones(len(u)),u,f(op.x*u)]; c=np.linalg.lstsq(fn(U),V,rcond=None)[0]; Z=fn(X)@c
 for extra in (False,True):
  a=np.c_[np.ones(len(X)),Z,X] if extra else np.c_[np.ones(len(X)),Z]; e=X[:,i]*X[:,j]; v=cv(a,e,.1,1); print('transfer',k,extra,v[0].round(4),v.max(0).round(4))
