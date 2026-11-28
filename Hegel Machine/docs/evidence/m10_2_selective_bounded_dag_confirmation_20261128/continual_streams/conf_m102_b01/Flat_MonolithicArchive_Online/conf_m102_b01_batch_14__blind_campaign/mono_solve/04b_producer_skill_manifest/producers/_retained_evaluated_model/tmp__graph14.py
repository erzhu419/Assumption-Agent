from explore14 import *
from scipy.optimize import root

def coords(s,x):
 a=s['panel_a'];u,v=a['u'],a['v'];d=aff(u);rows=np.arange(len(u))
 for _ in range(6):rows=np.flatnonzero(np.linalg.norm(d@ls(d[rows],v[rows])-v,axis=1)<2)
 cs=np.stack([ls(aff(np.c_[u[:,j],np.tanh(1.8*u[:,j]),np.sin(2.4*u[:,(j+2)%4])])[rows],v[rows,j]) for j in range(4)]).T
 bias,gain,sat,cross=cs
 z=np.stack([root(lambda q:bias+gain*q+sat*np.tanh(1.8*q)+cross*np.sin(2.4*np.roll(q,2))-row,(row-bias)/gain,tol=1e-11).x for row in x]); b=s['panel_b'];c=s['panel_c'];
 bf=lambda u:aff(np.cos(2.6*u));cf=lambda u:aff(np.tanh(1.75*u)+.30*np.sin(2.1*np.roll(u,1,axis=1)));df=lambda u:aff(np.c_[u,np.cos(1.28195*u)])
 w=bf(z)@ls(bf(b['u']),b['v']);c1=lambda u:cf(u)@ls(cf(c['u']),c['v1']);c2=lambda u:df(u)@ls(df(c['u']),c['v2']);
 return dict(x=x,z=z,w=w,q=c1(w),r=c2(w),qz=c1(z),rz=c2(z),rq=c2(c1(w)),qr=c1(c2(w)))

def yy(s):
 t=s['eval_times'];p=s['train']['traj'];
 def bb(r):return -np.expm1(-t[:,None]*r)
 def e(r):
  h=bb(r);return (h@ls(h,p.T)-p.T).ravel()
 rates=least_squares(e,[.4,1.8],bounds=([.02,1.],[.99,4.])).x
 print('rates',rates,np.std(e(rates)));return np.c_[s['train']['y'],ls(bb(rates),p.T).T]

if __name__=='__main__':
 from itertools import combinations
 x=s['train']['x'];ds=coords(s,x);y=yy(s);print('corr',np.corrcoef(y.T));rows=[];allb={}
 for k,u in ds.items():
  f={'id':u,'poly2':poly(u,[1,2]),'poly3':poly(u,[1,2,3]),'pair':poly(u,[2]),'cross':np.c_[u,u[:,0]*u[:,2],u[:,1]*u[:,3]],'asinh':np.arcsinh(2.15*u),'rat':u/(1+.22*u*u),'tanh':np.tanh(2.15*u),'sin':np.sin(2.15*u),'cos':np.cos(2.15*u),'asinh+id':np.c_[u,np.arcsinh(2.15*u)],'rat+id':np.c_[u,u/(1+.22*u*u)],'sqrt':np.sign(u)*np.sqrt(abs(u))}
  for k2 in ['z','w','q','r']:
   if k2!=k:
    f['+'+k2]=np.c_[u,ds[k2]];f['*'+k2]=np.c_[u,u*ds[k2]];f['*roll'+k2]=np.c_[u,u*np.roll(ds[k2],-1,axis=1)]
  for j,d in f.items():
   d=aff(d);allb[k,j]=d; e=d@ls(d[:52],y[:52])-y;rows.append((k,j,np.mean(abs(e[52:]),axis=0),np.std(e[:52],axis=0)))
 for j in range(3):print('target',j,*sorted(rows,key=lambda r:r[2][j])[:15],sep='\n')
 c=s['panel_c'];D=aff(np.c_[c['u'],np.cos(1.28195*c['u'])]); print('C2 weights',ls(D,c['v2'])); print('C2 residual',np.std(D@ls(D,c['v2'])-c['v2'],axis=0))
