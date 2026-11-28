from explore import *
from itertools import combinations
from scipy.linalg import qr
pa=s['panel_a']; ca=fit(poly(pa['u'],1),pa['v'])[0]; sc=np.diag(ca[1:]); pb=s['panel_b']; cb=ls(poly(pb['u'],1),pb['v'])
def coords(x):
 d={'raw':x,'cal':x*sc+ca[0],'inv':(x-ca[0])/sc}
 for k,z in list(d.items()):d[k+'B']=poly(z,1)@cb;d[k+'Binv']=(z-cb[0])@np.linalg.inv(cb[1:])
 return d
def feat(z,mode):
 if mode=='poly':return poly(z,2)
 if mode=='abs':return np.c_[poly(z,2),abs(z),z*abs(z)]
 if mode=='hinge':return np.c_[poly(z,2),abs(z),*[abs(z[:,i]+j*z[:,k]) for i,k in combinations(range(4),2) for j in [-1,1]]]
 if mode=='rational':return np.c_[poly(z,2),z/(1+abs(z)),z*z/(1+z*z)]
 if mode=='absprod':return poly(np.c_[z,abs(z)],2)
 if mode=='cubic':return np.c_[poly(z,2),z**3,z**4,*[z[:,i]**2*z[:,j] for i in range(4) for j in range(4) if i!=j]]
 if mode=='pairrat':return np.c_[poly(z,2),*[z[:,i]*z[:,j]/(1+.22*z[:,k]**2) for i,j in combinations_with_replacement(range(4),2) for k in range(4)]]
def scores(a,b):
 for r in [1e-10,.001,.03,.3]:
  aa=a[:52]; sd=np.sqrt((aa**2).mean(0)).clip(1e-12); aa=aa/sd
  w=np.linalg.solve(aa.T@aa+r*np.eye(aa.shape[1]),aa.T@b[:52])/sd[:,None]
  yield np.array([np.mean(abs(a[:52]@w-b[:52]),0)[0],np.mean(abs(a[52:]@w-b[52:]),0)[0],np.mean(abs(a[:52]@w-b[:52])[:,1:]),np.mean(abs(a[52:]@w-b[52:])[:,1:])])
if __name__=='__main__':
 b=np.c_[y,t]; out=[]
 for k,z in coords(x).items():
  for m in ['poly','abs','hinge','rational','absprod','cubic','pairrat']:
   for i,sc_ in enumerate(scores(feat(z,m),b)):out.append((k,m,i,sc_))
 for col in [1,3]:
  print('CV ranking',col);print(*sorted(out,key=lambda a:a[-1][col])[:15],sep='\n')
 for name in ['v1','v2']:
  u=s['panel_c']['u']; v=s['panel_c'][name]
  print('---', name);print('abs coeff',ls(np.c_[poly(u,1),abs(u)],v));print('quadratic coeff',ls(poly(u,2),v))
  print('hinge coeff',ls(feat(u,'hinge'),v))
 print('temporal coefficients, raw and inv'); r=np.array([.3762,1.8168,5.2872]); temp=-np.expm1(-ts[:,None]*r); amp=ls(temp,t.T).T
 print(np.c_[x,amp,y][:30]); print('amp std',amp.std(0));print('temporal errors',np.std(t-amp@temp.T,0)); print('B orthog',cb[1:].T@cb[1:]); print('B',cb)
