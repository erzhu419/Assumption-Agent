exec(open('/tmp/search11.py').read().split('base=[z,')[0])
from scipy.optimize import least_squares
p0=s['panel_a'];u=p0['u'];v=p0['v'];keep=np.max(abs(v),axis=1)<3;u,v=u[keep],v[keep];d=4
# Profile frequency with a phase-enabled sine and test the opposite-coordinate structure.
for kind in ('sin','phase','pair'):
 def des(z,w):
  extras={'sin':[],'phase':[np.cos(w*z)],'pair':[z*np.roll(z,2,axis=1),np.cos(w*z)]}[kind];return np.c_[aff(z),np.sin(w*z),*extras]
 def loss(w):
  a=des(u,w);return np.mean((a@fit(a,v)-v)**2)
 w=minimize_scalar(loss,bounds=(1.6,3.2),method='bounded').x;c=fit(des(u,w),v); zz=np.array([least_squares(lambda z0:des(z0[None],w)[0]@c-row,zi).x for zi,row in zip(z,x)]);print('cal',kind,w,np.sqrt(loss(w)),np.mean(abs(aff(zz)[n:]@fit(aff(zz)[:n],t[:n])-t[n:])))
a=m._features(s,z)[0];yp=a@fit(a[:n],y[:n]);print('residuals',np.round(y-yp,3));print('max',np.c_[z,y,y-yp][np.argsort(abs(y-yp))[-12:]].round(3));
# Train-only robust fits, compact smooth bases, and paired nonlinear regressors.
D={'endpoint':a,'z2':m._basis(z,2),'z3':m._basis(z,3),'zq1*2':np.c_[m._basis(z,2),groups['zq1*2']],'zr1':np.c_[a,groups['zr1']],'zr2*2':np.c_[aff(z),q,r,g,groups['zr2*2']]}
for k0 in fs:
 for w0 in (w1,w2):
  zz=fs[k0](w0*z);D[k0+str(w0)]=np.c_[a,zz*np.roll(z,1,axis=1),zz*np.roll(z,2,axis=1),zz*np.roll(z,-1,axis=1)]
for k0 in ('sin','tanh','ash','atan','erf','rat','soft'):
 for w0 in (w1,w2):
  for axis in (1,2,3):
   aa=z+np.roll(z,axis,axis=1);bb=z-np.roll(z,axis,axis=1);D[f'{k0}/{w0}/{axis}']=np.c_[a,fs[k0](w0*aa),fs[k0](w0*bb)]

def robust(a,i,y,alpha,delta):
 scale=np.maximum(a[i].std(0),1e-7);scale[0]=1;b=a[i]/scale; ridge=np.diag(np.r_[0,np.full(b.shape[1]-1,alpha)]); w=np.ones(len(i));
 for _ in range(12):
  c=np.linalg.solve((b.T*w)@b+ridge,(b.T*w)@y[i]);w=np.minimum(1,delta/np.maximum(abs(y[i]-b@c),1e-9));
 return c/scale
R=[]
for k0,a in D.items():
 for alpha in (.003,.05,.25,1.):
  for delta in (.012,.04,.12,100):
   sc=np.array([np.mean(abs(a[j]@robust(a,i,y,alpha,delta)-y[j])) for i,j in folds]);R.append((sc[1:].mean(),sc.max(),sc[0],k0,alpha,delta))
print('robust oof',sorted(R)[:10]);print('robust public',sorted(R,key=lambda a:a[2])[:10]);print('tradeoff',sorted((r for r in R if r[2]<.038),key=lambda a:a[1])[:10]);
