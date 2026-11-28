exec(open('/tmp/probe.py').read().split('m=old._panel_maps')[0])
from scipy.optimize import root,least_squares
from itertools import combinations
j=np.arange(4); op=(j+2)%4; u,v=a['u'],a['v']; good=np.linalg.norm(v-lin(u)@ls(lin(u),v),axis=1)<3
u,v=u[good],v[good]
def ad(z,p):return np.stack([np.ones_like(z),z,np.tanh(p[0]*z),np.sin(p[1]*z[:,op])],axis=-1)
def af(p,ret=False):
 d=ad(u,p); m=np.stack([ls(d[:,k],v[:,k]) for k in j]); return m if ret else (np.sum(d*m,axis=-1)-v).ravel()
p=least_squares(af,[1.8,2.4]).x; m=af(p,True)
def inv(q):return np.array([root(lambda zz:np.sum(ad(zz[None],p)[0]*m,axis=-1)-xx,xx).x for xx in q]);
z=inv(x); y=s['train']['y']; t=s['train']['traj']; n=52
print('trajectory singular values:',np.linalg.svd(t,compute_uv=False)); print('trajectory mean, endpoint range:',t.mean(0),t[:,-1].min(),t[:,-1].max())
D={ 'affine':lambda z,h:lin(z), 'poly3':lambda z,h:np.c_[lin(z),z**3], 'poly235':lambda z,h:np.c_[lin(z),z*z,z**3,z**5], 'ratlin':lambda z,h:np.c_[lin(z/(1+h*z*z)),z], 'atanlin':lambda z,h:np.c_[lin(np.arctan(h*z)),z], 'tanhlin':lambda z,h:np.c_[lin(np.tanh(h*z)),z], 'sinlin':lambda z,h:np.c_[lin(np.sin(h*z)),z], 'ratpure':lambda z,h:lin(z**3/(1+h*z*z)), 'ratmix':lambda z,h:lin((z**3-.1*z)/(1+h*z*z)), 'atancube':lambda z,h:np.c_[lin(np.arctan(h*z)),z,z**3], 'ratquad':lambda z,h:np.c_[lin(z/(1+h*z*z)),z,z*z] }
def calerr(q):
 d=lin(q); h=np.mean(abs(d[n:]@ls(d[:n],t[:n])-t[n:])); yy=np.mean(abs(sink._scalar_readout(q[:n],q[n:],y[:n])-y[n:]));return round(yy,5),round(h,5)
for k,q in [('x',x),('z',z),('z**3',z**3),('z**5',z**5),('A nonlinear',np.tanh(1.8*z)),('z rational',z/(1+1.25*z*z))]:print(k,calerr(q))
for panel,key in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=s[panel]['u'],s[panel][key];
 for name,d in D.items():
  def ob(h):
   a=d(u,h); e=1-np.sum(np.linalg.qr(a,mode='reduced')[0]**2,axis=1); return np.mean(((v-a@ls(a,v))/e[:,None])**2)
  h=minimize_scalar(ob,bounds=(.05,4.),method='bounded').x; a=d(u,h); b=ls(a,v); q=d(z,h)@b
  print(panel,key,name,round(h,4),round(ob(h)**.5,6),calerr(q))
