exec(open('/tmp/probe.py').read().split('funs=')[0])
from scipy.special import erf
from scipy.optimize import least_squares
u=s['panel_b']['u']; v=s['panel_b']['v']; T=np.c_[s['train']['y'],s['train']['traj']]
i,j=np.triu_indices(4,1)
def F(x):return np.c_[aff(x),x[:,i]*x[:,j]]
families={'exp':lambda x,b:-np.expm1(-b*x*x), 'rat':lambda x,b:b*x*x/(1+b*x*x), 'cos':lambda x,b:1-np.cos(b*x), 'tanh':lambda x,b:1-np.tanh(b*x)/np.where(abs(b*x)>1e-12,b*x,1), 'erf':lambda x,b:1-np.sqrt(np.pi)*erf(b*x)/(2*np.where(abs(b*x)>1e-12,b*x,1)), 'power':lambda x,b:abs(x)**b, 'genrat':lambda x,b,c:1-(1+b*x*x)**(-c), 'genexp':lambda x,b,c:-np.expm1(-b*abs(x)**c)}
for name,fn in families.items():
 wide=name.startswith('gen'); starts=[1.,2.] if wide else [1.]
 for extra in (False,True):
  def design(x,p):return np.c_[aff(x) if extra else np.ones(len(x)),x*fn(x,*p)]
  def fit(p):
   A=design(u,p); c=ls(A,v); return A@c-v,c,A
  opts=[least_squares(lambda p:fit(p)[0].ravel(),[b,1.5] if wide else [b],bounds=(.02,8)) for b in starts]; p=min(opts,key=lambda o:o.cost).x
  r,c,A=fit(p); H=np.sum(A*np.linalg.pinv(A).T,axis=1); zz=design(z,p)@c; f=F(zz); fy=ls(f[:52],T[:52,0]); fs=ls(aff(zz[:52]),T[:52,1:]); print(name,extra,p,'rms/loo',np.std(r),np.std(r/(1-H[:,None])),'Gram',np.linalg.svd(c[-4:],compute_uv=False),'MAE',np.mean(abs(f[52:]@fy-T[52:,0])),np.mean(abs(aff(zz[52:])@fs-T[52:,1:])))
  if name=='exp' and extra:
   print('readout',fy,'state coeff',fs,'bias',c[0],'ratio',np.sum(c[1:5]*c[-4:],axis=1)/np.sum(c[-4:]**2,axis=1))
