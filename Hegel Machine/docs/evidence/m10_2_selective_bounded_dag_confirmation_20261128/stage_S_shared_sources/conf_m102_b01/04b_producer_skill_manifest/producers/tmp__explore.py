import numpy as np,pickle
from scipy.optimize import least_squares, minimize_scalar
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; t=s['eval_times']; print('times',t)
np.set_printoptions(precision=5,suppress=True,linewidth=160)
def ls(X,Y): return np.linalg.lstsq(X,Y,rcond=None)[0]
def aff(X): return np.c_[np.ones(len(X)),X]
for tag in ['t','t2','sqrt','log','exp','one']:
 g={'t':t,'t2':t*t,'sqrt':np.sqrt(t),'log':np.log1p(t),'exp':1-np.exp(-t),'one':np.ones_like(t)}[tag];
 def fit(k,ret=False):
  A=np.c_[-np.expm1(-k*t),g]; B=ls(A,T.T); e=T.T-A@B;return (A,B) if ret else np.mean(e**2)
 r=minimize_scalar(fit,bounds=(.05,20),method='bounded');print('time',tag,r.x,np.sqrt(r.fun));
 if tag=='t': A,B=fit(r.x,True)
for k in ['v1','v2']:
 u=s['panel_c']['u']; v=s['panel_c'][k]; print('panel c',k)
 for d in [1,2,3]:
  X=PolynomialFeatures(d).fit_transform(u); bh=ls(X,v); print('poly',d,np.sqrt(np.mean((X@bh-v)**2,axis=0)))
  if d==2: print(bh)
 l=ls(aff(u),v); print('joint residual sv',np.linalg.svd(v-aff(u)@l,compute_uv=False))
 for fn in [np.tanh,np.sin,lambda z:z/(1+abs(z)),lambda z:np.arctan(z)]:
  vals=[]
  for j in range(4):
   z=aff(u); p=np.r_[l[:,j],1.,0.]; rr=least_squares(lambda p:p[-2]*fn(z@p[:-2])+p[-1]-v[:,j],p,max_nfev=400);vals.append(np.sqrt(np.mean(rr.fun**2)))
  print('ridge',fn,vals)
za=aff(s['panel_a']['u']); va=s['panel_a']['v']; L=ls(za,va); L=least_squares(lambda b:(za@b.reshape(5,4)-va).ravel(),L.ravel(),loss='soft_l1',f_scale=.01).x.reshape(5,4)
zb=aff(s['panel_b']['u']); M=ls(zb,s['panel_b']['v']); Z=(x-L[0])@np.linalg.inv(L[1:]); Y=np.c_[y,B.T]
print('target linear',ls(aff(Z),Y)); print('target cubic coeff',ls(PolynomialFeatures(3).fit_transform(Z),Y));
for k,Q in [('raw',x),('cal',Z),('b',aff(Z)@M)]:
 for j in range(4):
  a=Q[:,j]; F=np.c_[aff(Z),a*a,a**3,np.tanh(a),np.sin(a),np.abs(a),a/(1+a*a),np.maximum(a,0)]; C=ls(F[:52],Y[:52]); print(k,j,np.mean(abs(F[52:]@C-Y[52:]),axis=0));
