exec(open('/tmp/probe.py').read().split('m=old._panel_maps')[0])
from scipy.optimize import root
u,v=a['u'],a['v']
def design(z): return np.c_[lin(z),z*np.roll(z,-1,axis=1),z**3]
d=design(u); e=np.linalg.norm(v-d@ls(d,v),axis=1); keep=np.argsort(e)[:-3]; ba=ls(d[keep],v[keep]); print('A coefficient rows 1, linear, cyclic product, cubic:\n',ba.round(5));
ul,vl=b['u'],b['v']; y=s['train']['y'];t=s['train']['traj']
for mode,cols in [('full',np.arange(13)),('affine',np.arange(5)),('affine+pair',np.arange(9)),('affine+cubic',np.r_[0:5,9:13])]:
 aa=ls(d[keep][:,cols],v[keep]); z=np.stack([root(lambda zz:design(zz[None])[:,cols][0]@aa-xx,xx).x for xx in x]); print('A',mode,'fit RMSE',np.sqrt(np.mean((d[keep][:,cols]@aa-v[keep])**2)),'latent max',abs(z).max())
 for tag,D in [('affine',lambda z,a:lin(z)),('raw',lambda z,a:lin(z)[:,1:]),('rational+linear',lambda z,a:np.c_[lin(z/(1+a*z*z)),z])]:
  def obj(a):
   dd=D(ul,a); h=np.sum(np.linalg.qr(dd,mode='reduced')[0]**2,axis=1);return np.mean(((vl-dd@ls(dd,vl))/(1-h[:,None]))**2)
  alpha=minimize_scalar(obj,bounds=(.03,5),method='bounded').x; dd=D(ul,alpha); bb=ls(dd,vl); c=D(z,alpha)@bb if tag!='raw' else z; n=52; yp=sink._scalar_readout(c[:n],c[n:],y[:n]);tp=sink._trajectory(c[:n],c[n:],t[:n],s['eval_times']); print(tag,alpha,np.mean(abs(yp-y[n:])),np.mean(abs(tp-t[n:]))); 
  if mode=='full' and tag=='rational+linear': print('B coefficient rows 1, rational, linear:\n',bb.round(5))
