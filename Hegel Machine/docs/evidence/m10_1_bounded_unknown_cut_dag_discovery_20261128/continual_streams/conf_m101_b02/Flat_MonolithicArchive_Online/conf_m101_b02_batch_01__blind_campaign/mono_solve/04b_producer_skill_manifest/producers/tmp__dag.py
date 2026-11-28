from panels import *
from scipy.special import erf
from itertools import product
from sklearn.preprocessing import PolynomialFeatures

def des0(z):return np.c_[np.ones(len(z)),z]
def fit(z,v):return ls(des0(z),v)
def ab(s):
 u,v=(s['panel_a'][k] for k in ('u','v')); co=[]
 for i in range(4):
  a=des0(u[:,i]); b=fit(u[:,i],v[:,i]);
  for _ in range(8):
   idx=np.argsort(abs(a@b-v[:,i]))[:38];b=ls(a[idx],v[idx,i]);
  co.append(b)
 co=np.array(co);return co[:,0],co[:,1]
def make(s):
 a,b=ab(s);out={'A':lambda x:x*b+a,'I':lambda x:(x-a)/b,'X':lambda x:x}
 p=s['panel_c']
 for k,vn,r in [('oddexp','v1',1.8),('softplus','v2',2.2),('sqrteven','v2',.39)]:
  f=lambda x,k=k,r=r:des(x,k,r,'L' if k=='sqrteven' else ''); c=ls(f(p['u']),p[vn]);out[k]=lambda x,f=f,c=c:f(x)@c
 p=s['panel_b'];u=p['u'];v=p['v']
 for k,r in [('erf',1.414),('tanh',1.375),('sin',2.835)]:
  f=lambda x,k=k,r=r:des(x,k,r,'L');c=ls(f(u),v);out[k]=lambda x,f=f,c=c:f(x)@c
 return out

def score(g,iy,tar,pen):
 n=len(iy); v=np.setdiff1d(np.arange(len(tar)),iy); scale=np.std(g[iy],axis=0); scale=np.where(scale<1e-9,1,scale); a=(g-g[iy].mean(0))/scale; m=tar[iy].mean(0);b=a[iy]; K=b@b.T; pr=a[v]@b.T@np.linalg.solve(K+pen*np.eye(n),tar[iy]-m)+m; e=np.abs(pr-tar[v]);return np.r_[e[:,0].mean(),e[:,1:].mean()]

def feat(z,d=2):return PolynomialFeatures(d,include_bias=False).fit_transform(z)
def cross(x,y):return (x[:,:,None]*y[:,None,:]).reshape(len(x),-1)
if __name__=='__main__':
 x=s['train']['x']; tar=np.c_[s['train']['y'],s['train']['traj']]; fs=make(s);out=[]
 for ak,bk in product(('X','A','I'),('X','erf','tanh','sin')):
  a=fs[ak](x); b=fs[bk](a)
  for tag,z in [('a',a),('b',b)]:
   c=fs['oddexp'](z);d=fs['softplus'](z);e=fs['sqrteven'](z)
   for name,g in [('b',b),('b2',feat(b)),('bc',np.c_[b,c]),('bd',np.c_[b,d]),('bcd',np.c_[b,c,d]),('bc2',feat(np.c_[b,c])),('bcd_BC',np.c_[b,c,d,cross(b,c)]),('bcd_BD',np.c_[b,c,d,cross(b,d)]),('bcd_CD',np.c_[b,c,d,cross(c,d)]),('bce_CD',np.c_[b,c,e,cross(c,e)]),('bcd2',feat(np.c_[b,c,d])),('bce2',feat(np.c_[b,c,e])),('abce2',feat(np.c_[a,b,c,e])),('abc2',feat(np.c_[a,b,c])),('b2cde',np.c_[feat(b,3),c,d,e]),('ce2',feat(np.c_[c,e]))]:
    for pen in (.00003,.001,.03,.3):
     er=score(g,np.arange(52),tar,pen);out.append((*er,ak,bk,tag,name,pen))
 for i in (0,1):print('sorted',i,*sorted(out,key=lambda x:x[i])[:20],sep='\n')
 print('joint',*sorted(out,key=lambda x:x[0]+x[1])[:15],sep='\n')
 # B: determine whether the linear and nonlinear coefficient matrices share a mixing matrix.
 p=s['panel_b']; u,v=p['u'],p['v'];
 for k,r in [('erf',1.414),('tanh',1.375),('sin',2.835)]:
  b=ls(des(u,k,r,'L'),v); print(k, np.round(b,4));res=v-des(u,k,r,'L')@b;print('resid sv',np.linalg.svd(res,compute_uv=False))
