import pickle,numpy as np,itertools
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def lin(f,y):return np.linalg.lstsq(f,y,rcond=1e-9)[0]
def one(x):return np.c_[np.ones(len(x)),x]
p=s['panel_a']; keep=np.linalg.norm(p['v'],axis=1)<4; u,v=p['u'][keep],p['v'][keep]
P=PolynomialFeatures(3); F=P.fit_transform(u); names=P.get_feature_names_out(); W=lin(F,v)
for j in range(4):print('A cubic',j,[(names[i],round(W[i,j],4)) for i in np.argsort(-abs(W[:,j]))[:10]])
funcs={'sin':np.sin,'cos':np.cos,'tanh':np.tanh,'soft':lambda x:x/(1+abs(x)),'rat':lambda x:x/(1+x*x),'atan':np.arctan}
for stage in ['a','c']:
 if stage=='a': U,V=u,v; bases={'lin':one(U)}
 else:
  U,V=s['panel_c']['u'],s['panel_c']['v2']; bases={'const':one(U[:,:0]),'sq':one(U**2),'lin':one(U)}
 out=[]
 for bn,B in bases.items():
  for an,fn in funcs.items():
   def design(q):return np.c_[B,fn(q[0]*U)]
   def err(q):
    f=design(q);return (f@lin(f,V)-V).ravel()
   qs=[least_squares(err,[k],bounds=(.1,12),gtol=1e-12) for k in [.7,1.8,3.2,4.8,6.5,8.3,10.5]];q=min(qs,key=lambda x:np.linalg.norm(x.fun));out.append((np.std(q.fun),bn,an,q.x))
 print(stage,'univariate',sorted(out,key=lambda x:x[0])[:12])
# Compare dense calibration with one or two shared nonlinear frequencies, and expose coefficients.
for an,bn in itertools.product(['sin','tanh','rat'], repeat=2):
 def design(q):return one(np.c_[u,funcs[an](q[0]*u),funcs[bn](q[1]*u)])
 def err(q):
  f=design(q);return (f@lin(f,v)-v).ravel()
 q=min((least_squares(err,k,bounds=(.15,10),max_nfev=100) for k in [[1,2],[2,4],[4,2],[3,6],[6,3],[5,5]]),key=lambda x:np.linalg.norm(x.fun));print('A double',an,bn,np.std(q.fun),q.x)
 if np.std(q.fun)<.004: print(np.round(lin(design(q.x),v),4))
# Test v2 on parity-respecting pairwise sums and differences.
for stage,U,V in [('a',u,v),('c',s['panel_c']['u'],s['panel_c']['v2'])]:
 B=one(U if stage=='a' else U**2)
 for kind,fn in funcs.items():
  if stage=='c' and kind not in ('cos',):continue
  for shift in (1,2,3):
   for sign in (1,-1):
    def design(q):return np.c_[B,fn(q[0]*(U+sign*np.roll(U,shift,axis=1)))]
    def err(q):
     f=design(q);return (f@lin(f,V)-V).ravel()
    q=min((least_squares(err,[k],bounds=(.1,9)) for k in [1,2.5,4,6]),key=lambda x:np.linalg.norm(x.fun))
    if np.std(q.fun)<.035 or stage=='c':print('pair',stage,kind,shift,sign,np.std(q.fun),q.x)
