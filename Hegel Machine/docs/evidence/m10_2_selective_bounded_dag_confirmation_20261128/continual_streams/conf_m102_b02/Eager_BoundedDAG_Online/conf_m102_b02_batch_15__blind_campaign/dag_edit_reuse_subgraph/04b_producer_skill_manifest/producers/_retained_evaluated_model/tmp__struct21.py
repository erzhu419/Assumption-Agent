from lab21 import *
from scipy.linalg import qr
from scipy.optimize import minimize_scalar
x=np.r_[tr['x'],s['test']['x']]; a,w=fit_a(); z=invert(x,a,w)
print('inversion residual',np.max(abs(z*w[1]+np.sin(a*z)*w[2]+np.sin(a*np.roll(z,2,axis=1))*w[3]+w[0]-x),axis=0)); print('panel lengths',[len(p['u']) for p in [A,B,C]])
# Separate even and odd terms with two independently profiled frequencies.
u,v=C['u'],C['v2']; opts={}
for n1,h in [('sin',np.sin),('tanh',np.tanh),('rat',lambda x:x/(1+x*x)),('asinh',np.arcsinh)]:
 for n2,g in [('cos',np.cos),('tanh2',lambda x:np.tanh(x)**2),('rat2',lambda x:x*x/(1+x*x)),('linear2',lambda x:x*x),('tanh_sq',lambda x:np.tanh(x*x))]:
  def fit(u,q):return D(h(q[0]*u),g(q[1]*u))
  def err(q):
   f=fit(u,q);return (f@reg(f,v)-v).ravel()
  o=least_squares(err,[.4,.4],bounds=(.015,4)); opts[n1,n2]=(rms(o.fun),o.x)
print('C2 forms', sorted(opts.items(),key=lambda q:q[1][0])[:8]);
f=D(z,z*z); print('Y ranges',Y.min(0),Y.max(0)); print('X ranges',x.min(0),x.max(0)); print('Y first',np.round(np.c_[z[:72],Y[:,[0,-1]]][:12],4));
T=Y[:,1:]; tm=s['eval_times']; print('times',tm); E=lambda p:-np.expm1(-tm[:,None]*p); o=least_squares(lambda p:(E(p)@reg(E(p),T.T)-T.T).ravel(),[2.15,.22],bounds=([1,.05],[6,1])); print('rates',o.x,rms(o.fun)); W=np.c_[Y[:,0],reg(E(o.x),T.T).T];
# Test families of sparse supervised bases, sharing functions but not coefficients.
cos=np.cos(2.6*z); co=D(np.cos(2.6*B['u'])); b=D(cos)@reg(co,B['v']); p=cyc(z); v1=D(p)@reg(D(cyc(C['u'])),C['v1']); v2=D(np.tanh(.21647*z),np.tanh(.21647*z)**2)@reg(D(np.tanh(.21647*u),np.tanh(.21647*u)**2),v);
np.savez('/tmp/struct21.npz',x=x,z=z,b=b,p=p,v1=v1,v2=v2,Y=Y,W=W);
def basis(h):
 return [('raw',h),('poly',polys(h,3)),('harmonics',np.c_[h,h*h,*[fun(a*h) for a in [.22,.5,1.,1.25,2.15,2.6,3.5,4.3,5.2,6.5,8.6,10.4] for fun in [np.sin,np.cos,np.tanh]]]),('poly2',polys(h,2))]
from sklearn.linear_model import OrthogonalMatchingPursuit, ElasticNet
res=[]
for name,h in [('z',z),('x',x),('b',b),('cos',cos),('v1',v1),('v2',v2),('zb',np.c_[z,b]),('mix+',z+b),('mix-',z-b),('all',np.c_[z,b,v1,v2]),('mixa',np.c_[z+b,z-b]),('h',np.c_[z,b,p,v1,v2,z+b,z-b,b+v1,b-v1])]:
 for nn,f in basis(h):
  if f.shape[1]>1500:continue
  f=f[:72];mu=f[:52].mean(0);sd=f[:52].std(0); f=(f-mu)/np.maximum(sd,1e-8)
  for k in [4,8,12,20,32]:
   if k>f.shape[1]:continue
   model=OrthogonalMatchingPursuit(n_nonzero_coefs=k).fit(f[:52],W[:52]); e=np.abs(model.predict(f[52:])-W[52:]).mean(0); res.append((tuple(e),name,nn,k));
for j in range(3):print('sparse head',j,sorted(res,key=lambda p:p[0][j])[:8]);
