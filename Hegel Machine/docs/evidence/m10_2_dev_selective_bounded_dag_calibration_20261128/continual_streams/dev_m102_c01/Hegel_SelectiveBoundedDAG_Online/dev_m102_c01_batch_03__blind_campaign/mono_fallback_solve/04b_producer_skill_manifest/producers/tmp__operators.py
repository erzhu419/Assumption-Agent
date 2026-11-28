import numpy as np,pickle
from scipy.optimize import minimize_scalar
from scipy.linalg import solve
from scipy.spatial.distance import cdist
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; Y=np.c_[s['train']['y'],s['train']['traj']]; n=len(x)
def affine(u,v):
 X=np.c_[np.ones(len(u)),u]; w=np.linalg.lstsq(X,v,rcond=None)[0]; return w
p=s['panel_a']; u,v=p['u'],p['v']; keep=np.ones(len(u),bool)
for j in range(6):
 a=affine(u[keep],v[keep]); e=np.linalg.norm(np.c_[np.ones(len(u)),u]@a-v,axis=1); keep=np.argsort(e)[:44]; print('A',j,np.round(np.sort(e)[[0,22,43,44,45,46,47]],6))
b=affine(s['panel_b']['u'],s['panel_b']['v']);print('A map',a);print('B map',b)
from scipy.optimize import least_squares
p=s['panel_c']; u=p['u']; v=p['v1'];
F=lambda z: np.c_[np.ones(len(z)),z]
def profile(f,v=v):
 w=np.linalg.lstsq(f,v,rcond=None)[0]; return f@w-v
ops={}
for typ in ['tanh','tanhplus','softplus','sinplus','tanhshift','tanhshiftplus','soft','atan','atanplus']:
 def feature(z,h):
  a=z*h[:4]+(h[4:] if 'shift' in typ else 0)
  if 'tanh' in typ:t=np.tanh(a)
  if 'soft' in typ:t=a/np.sqrt(1+a*a)
  if 'sin' in typ:t=np.sin(a)
  if 'atan' in typ:t=np.arctan(a)
  return np.c_[F(z) if 'plus' in typ else np.ones((len(z),1)),t]
 h=least_squares(lambda h:profile(feature(u,h)).ravel(),np.r_[np.full(4,1.5),np.zeros(4) if 'shift' in typ else []],max_nfev=150,bounds=(np.r_[np.full(4,.02),np.full(4,-2) if 'shift' in typ else []],3.8)).x
 w=np.linalg.lstsq(feature(u,h),v,rcond=None)[0]; err=np.sqrt(np.mean(profile(feature(u,h))**2)); print(typ,err,h); ops[typ]=(lambda z,f=feature,h=h,w=w:f(z,h)@w) if False else (h,w)
# Test componentwise mixtures of the denominator and numerator response and an affine outer readout.
for typ in ['tanhsq','tanhmixed','ratodd','gelu','erf','arctan','softr','tanhrot']:
 if typ=='tanhrot':
  for j in range(4):
   def func(p):return p[0]+p[1]*np.tanh(u@p[2:6]+p[6])+u@p[7:]
   h=least_squares(lambda p:func(p)-v[:,j],np.r_[0,1,affine(u,v)[1:,j],0,np.zeros(4)],max_nfev=500).x; print('rot',j,np.sqrt(np.mean((func(h)-v[:,j])**2)),h)
  continue
 from scipy.special import erf
 def feat(z,h):
  a=z*h[:4]; fun={'tanhsq':lambda:np.tanh(a*a),'tanhmixed':lambda:np.tanh(a),'ratodd':lambda:a/(1+a*a),'gelu':lambda:a*erf(a),'erf':lambda:erf(a),'arctan':lambda:np.arctan(a),'softr':lambda:a/np.sqrt(1+a*a)}[typ]; return np.c_[F(z),fun(),np.tanh(a)**2] if typ in ('tanhsq','tanhmixed') else np.c_[F(z),fun()]
 h=least_squares(lambda h:profile(feat(u,h)).ravel(),np.full(4,1.3),max_nfev=150,bounds=(.02,4.)).x;print(typ,np.sqrt(np.mean(profile(feat(u,h))**2)),h)
def f1(z):
 h,w=ops['tanhshiftplus'];return np.c_[F(z),np.tanh(z*h[:4]+h[4:])]@w
def f2(z):return np.c_[F(z),z*z/(1+.1625*z*z)]@np.linalg.lstsq(np.c_[F(u),u*u/(1+.1625*u*u)],p['v2'],rcond=None)[0]
A=lambda z: (z-a[0])/np.diag(a[1:]); B=lambda z:z@b[1:]+b[0]
coords={'x':x,'a':A(x),'ab':B(A(x)),'b':B(x)}
for k,z in list(coords.items()):
 coords[k+'1']=f1(z);coords[k+'2']=f2(z);coords[k+'1b']=B(f1(z));coords[k+'2b']=B(f2(z))
rng=np.random.default_rng(456);folds=[(np.arange(52),np.arange(52,72))];
for _ in range(10):
 i=rng.permutation(n);folds.append((i[:52],i[52:]))
def check(K,r,folds=folds):
 E=[]
 for i,j in folds:
  e=abs(K[np.ix_(j,i)]@solve(K[np.ix_(i,i)]+r*np.eye(len(i)),Y[i],assume_a='pos')-Y[j]);E.append([e[:,0].mean(),e[:,1:].mean()]);
 E=np.array(E);return (E[:,0].mean(),E[0,0],E[:,0].max(),E[0,1],E[:,1].max())
if __name__=='__main__':
 results=[]
 for k,z in coords.items():
  z=(z-z.mean(0));z/=np.sqrt((z*z).sum(1).mean()); D=cdist(z,z,'sqeuclidean'); L=z@z.T
  for q in [0,1,2,3]:
   for h in [.15,.35,.65,1.2]:
    for wgt in [0,.3,1]:
     K=(1+L)**q+wgt*np.exp(-h*D)
     for r in [2e-5,.0003,.003]:results.append((*check(K,r),k,q,h,wgt,r))
 print('BEST',*sorted(results)[:18],sep='\n');print('BEST GATE',*sorted(results,key=lambda e:e[1])[:12],sep='\n')
