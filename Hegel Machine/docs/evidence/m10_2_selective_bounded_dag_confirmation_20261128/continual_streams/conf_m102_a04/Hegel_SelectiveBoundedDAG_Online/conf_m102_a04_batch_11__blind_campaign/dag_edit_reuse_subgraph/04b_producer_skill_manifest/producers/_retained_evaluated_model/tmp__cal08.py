import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=s['panel_a']['u']; v=s['panel_a']['v']; keep=np.linalg.norm(v,axis=1)<4; u,v=u[keep],v[keep]; w=np.roll(u,-2,axis=1); ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
fn={'rational':lambda x,q:x/(1+q*x*x),'tanh':lambda x,q:np.tanh(q*x)/q,'sin':lambda x,q:np.sin(q*x)/q,'atan':lambda x,q:np.arctan(q*x)/q,'softsign':lambda x,q:x/(1+q*abs(x)),'cubic':lambda x,q:x**3,'quadratic':lambda x,q:x*x}
for f,g in [(f,g) for f in fn for g in ['rational','sin','tanh','atan','softsign','cubic']]:
 def design(q):
  return [np.c_[np.ones(len(u)),a,fn[f](a,q[0]),fn[g](b,q[1])] for a,b in zip(u.T,w.T)]
 def residual(q):return np.column_stack([a@ls(a,b)-b for a,b in zip(design(q),v.T)]).ravel()
 q=least_squares(residual,[.9,.9],bounds=([.005]*2,[6.]*2),gtol=1e-11,ftol=1e-11).x
 score=np.std(residual(q)); print('%-9s %-9s %.8f'%(f,g,score),q, np.array([ls(a,b) for a,b in zip(design(q),v.T)]).round(4).tolist() if score<.006 else '')
for name,key in [('panel_b','v'),('panel_c','v1')]:
 p=s[name]; a=p['u']; t=p[key]
 def design(q):return np.c_[np.ones(len(a)),a/(1+q[0]*np.abs(a))+q[1]*a]
 e=lambda q:(design(q)@ls(design(q),t)-t).ravel()
 q=least_squares(e,[.8,-.15],bounds=([.01,-2],[10,2]),gtol=1e-13).x
 print(name,key,q,np.std(e(q)), np.max(abs(e(q))))
