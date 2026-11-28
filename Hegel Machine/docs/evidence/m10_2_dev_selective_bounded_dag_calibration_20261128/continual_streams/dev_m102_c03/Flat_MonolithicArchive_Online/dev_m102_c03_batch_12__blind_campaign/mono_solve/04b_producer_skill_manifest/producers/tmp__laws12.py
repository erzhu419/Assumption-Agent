import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def ls(a,b): return np.linalg.lstsq(a,b,rcond=1e-9)[0]
def aff(*x): return np.column_stack((np.ones(len(x[0])),*x))
def mon(x,k): return np.column_stack([np.prod(x[:,j],axis=1) for j in combinations_with_replacement(range(4),k)])
def caldesign(x,j,kind):
 w=x[:,j]; p=x[:,(j+1)%4]; m=x[:,(j-1)%4]; q=x[:,(j+2)%4]
 d={'diag':(w,), 'diag3':(w,w*w,w**3),'aff':(x,), 'cyc+':(w,p,w*p,w**3), 'cyc-':(w,m,w*m,w**3), 'cyc2':(w,q,w*q,w**3),'mix':(x,w*w,w**3,w*p,w*m,w*q), 'quad3':(x,mon(x,2),w**3),'full3':(x,mon(x,2),mon(x,3))}
 return aff(*d[kind])
def calfit(kind):
 u,v=s['panel_a']['u'],s['panel_a']['v']; a=[caldesign(u,j,kind) for j in range(4)]
 keep=np.ones(len(u),bool)
 # Initialize inliers from robust affine approximation.
 for _ in range(5):
  z=aff(u); r=np.linalg.norm(z@ls(z[keep],v[keep])-v,axis=1); keep=r<max(.004,3*np.median(r))
 for _ in range(4):
  c=[ls(aj[keep],v[keep,j]) for j,aj in enumerate(a)]; r=np.linalg.norm(np.column_stack([aj@b for aj,b in zip(a,c)])-v,axis=1); keep=r<max(.004,4*np.median(r))
 return c,r,keep
fs={'linear':lambda x,k:aff(x),'cyc':lambda x,k:aff(x*np.roll(x,-1,axis=1)),'quad':lambda x,k:aff(mon(x,2)),'linear_quad':lambda x,k:aff(x,mon(x,2)),'tanh':lambda x,k:aff(np.tanh(x*k)),'tanh_lin':lambda x,k:aff(np.tanh(x*k),x),'sine':lambda x,k:aff(np.sin(x*k)),'sine_lin':lambda x,k:aff(np.sin(x*k),x),'cos':lambda x,k:aff(np.cos(x*k)),'cos_lin':lambda x,k:aff(np.cos(x*k),x),'soft':lambda x,k:aff(x/(1+np.abs(x*k))),'soft_lin':lambda x,k:aff(x/(1+np.abs(x*k)),x),'rational':lambda x,k:aff(x/(1+(x*k)**2)),'evenrational':lambda x,k:aff(1/(1+(x*k)**2)),'rbf':lambda x,k:aff(np.exp(-(x*k)**2)),'square':lambda x,k:aff(x**2),'mixcos':lambda x,k:aff(np.cos(x*k),x**2),'sincos':lambda x,k:aff(np.sin(x*k),np.cos(x*k))}
def branch(u,v,f):
 def res(k):
  a=f(u,k); return (a@ls(a,v)-v).ravel()
 sol=min((least_squares(res,[k],bounds=(.05,8),max_nfev=80) for k in [.55,1.25,2.15,3.5]),key=lambda a:np.mean(a.fun**2)); k=sol.x
 a=f(u,k); q=np.linalg.svd(a,full_matrices=False)[0]; loo=sol.fun.reshape(v.shape)/np.maximum(.01,1-(q*q).sum(1))[:,None]
 return np.sqrt(np.mean(sol.fun**2)),np.sqrt(np.mean(loo**2)),k[0]
if __name__=='__main__':
 for k in ['diag','diag3','aff','cyc+','cyc-','cyc2','mix','quad3','full3']:
  c,r,g=calfit(k);print('cal',k,np.round(np.quantile(r,[0,.5,.75,.8,.9,1]),6),g.sum(),np.round(c,4) if k in ('diag3','cyc+','cyc-') else '')
 for p,j in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
  u,v=s[p]['u'],s[p][j]; print(p,j); rr=sorted((branch(u,v,f),name) for name,f in fs.items()); print([(name,np.round(a,6).tolist()) for a,name in rr])
