import numpy as np,pickle,importlib.util,itertools
from scipy.optimize import least_squares
sp=importlib.util.spec_from_file_location('f','/holdout_task/_frozen_nodes/node_00.py'); f=importlib.util.module_from_spec(sp);sp.loader.exec_module(f)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; out=f.run_node({'__input__':s}); tr=s['train']; y=tr['y']; T=tr['traj']; t=s['eval_times'];
def poly(x,p):
 terms=[i for d in range(p+1) for i in itertools.combinations_with_replacement(range(x.shape[1]),d)];return np.stack([np.prod(x[:,i],axis=1) for i in terms],1),terms
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def err(a,b):return np.mean(abs(a-b))
u=s['panel_b']['u'];v=s['panel_b']['v'];a=f._affine(u); z=out['z']; zt=out['zt']; raw=np.linalg.solve(ls(a,v)[1:].T,(z-ls(a,v)[0]).T).T
from scipy.optimize import least_squares
from scipy.special import erf
F={'sin':np.sin,'tanh':np.tanh,'erf':erf,'rat':lambda a:a/(1+a*a),'gau':lambda a:a*np.exp(-a*a),'soft':lambda a:a/np.sqrt(1+a*a)}
U=u; V=v
for name,fn in F.items():
 def D(X,h):return np.c_[np.ones(len(X)),X,fn(h*X)]
 def resid(p):
  A=D(U,p[0]); return (A@ls(A,V)-V).ravel()
 h=min([least_squares(resid,[h],bounds=(.05,6)) for h in [.8,1.5,2.8]],key=lambda o:o.cost).x[0];A=D(U,h);C=ls(A,V); E=A@C-V;print(name,round(h,5),np.linalg.norm(E),np.linalg.norm(E,axis=1).round(4))
P=np.load('/tmp/aligned.npz'); X=P['X'];
for name,xx in [('aligned',X),('raw',raw)]:
 for nonlinear in ['sin','tanh','erf','rat','soft','gau','cos','both','raw']:
  def feat(X,p):
   h=p[0]; fn=F.get(nonlinear,np.sin); W=fn(h*X); Z=np.c_[np.ones(len(X)),X,W] if nonlinear not in ('raw','cos') else np.c_[np.ones(len(X)),X] if nonlinear=='raw' else np.c_[np.ones(len(X)),X,np.cos(h*X)];
   return np.c_[Z,poly(X,2)[0][:,5:],np.cos(h*X)] if nonlinear=='both' else np.c_[Z,poly(X,2)[0][:,5:]]
  for q,B in [('y',y),('T',T@np.linalg.svd(T,full_matrices=False)[2][0])]:
   def res(p):
    A=feat(xx[:52],p);return A@ls(A,B[:52])-B[:52]
   best=min([least_squares(res,[h],bounds=(.1,14),max_nfev=80) for h in [0.8,2,4,6,9,12]],key=lambda o:o.cost); A=feat(xx,best.x);r=A@ls(A[:52],B[:52])-B;print(name,nonlinear,q,best.x.round(3),np.mean(abs(r[:52])).round(4),np.mean(abs(r[52:])).round(4))
