import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];p=s['panel_c'];u=p['u'];print('panel_c',u.shape,u.min(0),u.max(0));rng=np.random.default_rng(53);ix=rng.permutation(len(u));a,b=ix[:len(ix)*3//4],ix[len(ix)*3//4:]
def ls(d,v):return np.linalg.lstsq(d,v,rcond=None)[0]
for k in ['v1','v2']:
 v=p[k]; print(k)
 def report(name,d):print(name,d.shape[1], 'RMS full, held',np.mean((d@ls(d,v)-v)**2)**.5,np.mean((d[b]@ls(d[a],v[a])-v[b])**2)**.5)
 for dg in range(1,5):
  report('separable poly '+str(dg),np.column_stack([np.ones(len(u))]+[u**i for i in range(1,dg+1)])); report('full poly '+str(dg),np.column_stack([np.ones(len(u))]+[np.prod(u[:,ii],axis=1) for i in range(1,dg+1) for ii in itertools.combinations_with_replacement(range(4),i)]))
 for name,fn in {'soft':lambda z:z/(1+abs(z)),'tanh':np.tanh,'rational':lambda z:1/(1+z*z),'rational_odd':lambda z:z/(1+z*z),'cos':np.cos,'sin':np.sin,'exp_neg_square':lambda z:np.exp(-z*z),'abs':np.abs,'softplus':lambda z:np.logaddexp(0,z),'exp':np.exp,'arctan':np.arctan,'sqrt':lambda z:np.sqrt(1+z*z)}.items():
  def D(q):return np.c_[np.ones(len(u)),u,fn(q[0]*u+q[1:])]
  def R(q):
   d=D(q);return (d[a]@ls(d[a],v[a])-v[a]).ravel()
  opts=[least_squares(R,[i,0,0,0,0],bounds=([.03,-4,-4,-4,-4],[8,4,4,4,4]),max_nfev=130) for i in [.45,1.25,2.5]]
  opt=min(opts,key=lambda o:o.cost); print(name,opt.x.round(4));report('fit',D(opt.x))
