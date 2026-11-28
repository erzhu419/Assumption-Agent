import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(suppress=True,precision=4,linewidth=150)
F=lambda x:np.c_[np.ones(len(x)),x]
L=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
u,v=[s['panel_a'][i] for i in ('u','v')]; i,j=np.triu_indices(len(u),1); du=u[j]-u[i]; g=np.nanmedian(np.where(abs(du)>.2,(v[j]-v[i])/np.where(du==0,1,du),np.nan),0); a=np.median(v-u*g,0); r=np.max(abs(v-u*g-a),1); keep=r<2; print('A',a,g,keep.sum(),np.sort(r)); M=L(F(u[keep]),v[keep]); print('A full inlier',M,'resid',np.std(v[keep]-F(u[keep])@M,0)); x=(s['train']['x']-M[0])@np.linalg.inv(M[1:]); Y=np.c_[s['train']['y'],s['train']['traj']];
def score(z):
 r=Y-z@L(z[:52],Y[:52]); return (np.mean(abs(r[52:,0]),0),np.mean(abs(r[52:,1:])),np.mean(abs(r[:52,1:])))
for k,l in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u=s[k]['u']; v=s[k][l]; m=L(F(u),v); print(k,l,'linear svals',np.linalg.svd(m[1:],compute_uv=False),'r singular',np.linalg.svd(v-F(u)@m,compute_uv=False)); print('col 0 u/v/res',np.c_[u,v[:,0],(v-F(u)@m)[:,0]])
 bases={'lin':lambda u:u,'quad':lambda u:np.c_[u,u*u], 'abs':lambda u:np.c_[u,abs(u)],'tanh1':lambda u:np.tanh(u),'sin1':np.sin,'sin2':lambda u:np.sin(2*u),'tanhadd':lambda u:np.c_[u,np.tanh(u)],'cosadd':lambda u:np.c_[u,np.cos(u)], 'cross':lambda u:np.c_[u,u*u], 'radial':lambda u:np.c_[u,u*np.sum(u*u,axis=1,keepdims=True)]}
 fits=[]
 for bn,f in bases.items():
  U=F(f(u)); H=np.sum(np.linalg.qr(U)[0]**2,1); b=L(U,v); rr=(v-U@b)/(1-H[:,None]); q=F(f(x))@b; fits.append((np.mean(abs(rr)),bn,score(F(q)),score(F(np.c_[x,q,q*q,np.tanh(q)]))))
 for z in sorted(fits): print(z)
 for trans in ['tanh','sin','asinh','cubic','clip','softsign']:
  def h(z):
   if trans=='tanh':return np.tanh(z)
   if trans=='sin':return np.sin(z)
   if trans=='asinh':return np.arcsinh(z)
   if trans=='cubic':return z**3
   if trans=='clip':return np.clip(z,-1,1)
   return z/(1+abs(z))
  U=F(u)
  for form in [0,1]:
   def model(w,U):
    z=U@w[:20].reshape(5,4); return z+w[20:]*h(z) if form==0 else (w[20:]*h(z))
   b=least_squares(lambda w:(model(w,U)-v).ravel(),np.r_[m.ravel(),np.full(4,.1 if form==0 else 1.)],max_nfev=200).x
   q=model(b,F(x)); print(trans,form,'panel MAE',np.mean(abs(model(b,U)-v)),score(F(np.c_[x,q,q*q])))
 t=s['eval_times']
print('time normalized top V',np.linalg.svd(Y[:52,1:],full_matrices=False)[2][:4]);
for mode in ['exp','exp+1','exp+t','2exp','2exp+t','2exp+1','3exp']:
 n=2 if mode.startswith('2') else 3 if mode.startswith('3') else 1
 def base(r):
  z=-np.expm1(-np.exp(r[:,None])*t); return np.r_[z,t[None,:]] if mode.endswith('+t') else np.r_[z,np.ones((1,len(t)))] if mode.endswith('+1') else z
 def fun(r):
  b=base(r); yy=Y[:52,1:]; return (yy-yy@np.linalg.pinv(b)@b).ravel()
 fit=least_squares(fun,np.linspace(-1,1,n),bounds=(-7,4)); print(mode,np.exp(fit.x),np.mean(abs(fun(fit.x))))
