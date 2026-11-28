import numpy as np,pickle,runpy
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; m=runpy.run_path('/holdout_task/node_01.py'); up=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':s}); U,V=s['panel_a']['u'],s['panel_a']['v']; good=np.linalg.norm(V-U,axis=1)<2.; U,V=U[good],V[good]; t=s['train']['traj']; y=s['train']['y']; xs=s['train']['x']; ls=m['_ls']; rot=lambda q:np.roll(q,-2,axis=1)
def design(q,kind,p):
 z=rot(q); k=float(s['telemetry_vec'][0]); hh={'sin':lambda a:np.sin(p*a),'sin2':lambda a:np.sin(p*a),'tanh':lambda a:np.tanh(p*a),'rat':lambda a:a/(1+p*a*a),'cubic':lambda a:a**3,'sinplus':lambda a:np.sin(k*a),'ratplus':lambda a:a/(1+p*a*a)}[kind]; ff=[np.ones_like(q),q,hh(z)];
 if kind in ('sin2','sinplus','ratplus','cubic'): ff.append(z)
 if kind in ('sinplus','ratplus'): ff.extend([q*q,q**3,z*z,q*z]);
 return np.stack(ff,axis=-1)
def fit(kind,p):
 a=design(U,kind,p); w=np.array([ls(a[:,j],V[:,j]) for j in range(4)]); return w,(a*w).sum(-1)-V
splits=[(np.arange(52),np.arange(52,len(t)))]+[(np.setdiff1d(np.arange(len(t)),i),i) for i in np.array_split(np.random.default_rng(12).permutation(len(t)),5)]
def eval(q):
 b=m['_panel_features'](q,up); inv=1/(1+s['telemetry_vec'][3]*q*q); u=s['panel_c']['u']; c=np.c_[m['_design'](q,15),b]@ls(np.c_[m['_design'](u,15),m['_panel_features'](u,up)],np.c_[s['panel_c']['v1'],s['panel_c']['v2']]); co=b*np.roll(q,-1,axis=1); result=[]
 for X,Y,a in [(np.c_[b,c,inv],y,.025),(np.c_[b,inv,q**3],t,.25),(np.c_[b,co,c,inv],t,.025),(np.c_[b,co.sum(1)],t,1e-6)]:
  e=np.array([np.mean(abs(m['_ridge'](X[tr],Y[tr],X[te],a)-Y[te])) for tr,te in splits]); result.append(np.round([e[0],e[1:].mean(),e[1:].max()],5))
 return result
q0=m['_invert'](xs,m['_fit_calibration'](s['panel_a'])); print('baseline',eval(q0));
for kind in ['sin','sin2','tanh','rat','cubic','sinplus','ratplus']:
 p=least_squares(lambda p:fit(kind,p[0])[1].ravel(),[2.15 if kind.startswith('sin') else 1.],bounds=(.1,6.)).x[0]; w,r=fit(kind,p); print(kind,p,'cal rmse',np.sqrt(np.mean(r*r)),'w',w.round(4).tolist()); print('resid corr',np.round(np.corrcoef(np.c_[r,U].T)[:4,4:],2).tolist()); q=np.array([least_squares(lambda a:(design(a[None,:],kind,p)[0]*w).sum(-1)-x,z,max_nfev=35).x for x,z in zip(xs,q0)]); print('pred',eval(q))
