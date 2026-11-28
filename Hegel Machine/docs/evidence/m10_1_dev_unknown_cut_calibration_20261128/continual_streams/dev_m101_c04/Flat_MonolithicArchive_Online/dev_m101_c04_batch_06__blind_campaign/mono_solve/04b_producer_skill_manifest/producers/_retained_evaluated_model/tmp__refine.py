import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=s['train']['y']; T=s['train']['traj']; d=np.load('/tmp/stages.npz'); z=d['b']; t=s['eval_times']; pairs=np.array(list(itertools.combinations_with_replacement(range(4),2))); ii,jj=pairs.T
U=np.c_[np.ones(len(z)),z]; Q=np.prod(z[:,pairs],axis=2); F=np.c_[U,Q]
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def mat(w):
 m=np.zeros((4,4)); m[ii,jj]=m[jj,ii]=w/(1+(ii!=jj)); return m
def mae(x):return np.mean(abs(x))
def score(i,j,a):
 w=ls(a[i],y[i]); return np.array([mae(a[i]@w-y[i]),mae(a[j]@w-y[j])])
rng=np.random.default_rng(9); folds=[(np.arange(56),np.arange(56,72))]+[(np.setdiff1d(np.arange(72),j),j) for j in np.array_split(rng.permutation(72),5)]
for label in ['v1','v2','both']:
 for coord in ('z','b','x'):
  c=[d[key] for key in d.files if (key.startswith(label) or label=='both') and ('_'+coord+'_') in key and (key.endswith('oddpower1') or key.endswith('softabs1'))]; a=np.c_[F,*c]; print(label,coord,np.round([score(i,j,a) for i,j in folds],4).tolist())
for k,(i,j) in enumerate(folds):
 u=U[i]; q=Q[i]; wq=ls(q-u@ls(u,q),T[i]-u@ls(u,T[i])); _,_,r=np.linalg.svd(wq,full_matrices=False); m=mat(wq@r[0]); ev,vec=np.linalg.eigh(m); m2=(vec*np.repeat(ev.reshape(2,2).mean(1),2))@vec.T; st=[]
 for a in [F,np.c_[U,np.einsum('ni,ij,nj->n',z,m,z)],np.c_[U,np.einsum('ni,ij,nj->n',z,m2,z)]]:
  st.append(mae(a[j]@ls(a[i],T[i])-T[j]))
 print('state',k,np.round(st,6)); p0=ls(F[i],y[i]); direction=p0[1:5]/np.linalg.norm(p0[1:5]);
 for name in ['rat','tanh','blend','shiftblend','twogate','parity']:
  def basis(p):
   a=U@p[:5]; rat=a/(1+a*a)
   if name=='rat':return np.c_[F,rat]
   if name=='tanh':return np.c_[F,np.tanh(a)]
   if name=='blend':return np.c_[F,rat,np.tanh(p[5]*a)]
   if name=='shiftblend':return np.c_[F,rat,np.tanh(p[5]*a+p[6])]
   if name=='twogate':return np.c_[F,rat,np.tanh(U@p[5:])]
   return np.c_[F,rat,1/(1+a*a)]
  def res(p):
   a=basis(p)[i];return a@ls(a,y[i])-y[i]
  init=[[]] if name in ['rat','tanh','parity'] else {'blend':[[1.5],[.7],[2]],'shiftblend':[[1.5,0],[.7,.3],[2,-.3]],'twogate':[[0,*direction*v] for v in [1.5,.6,2.5]]}[name]; fit=min((least_squares(res,np.r_[0,direction*scale,b],max_nfev=100) for b in init for scale in [.85,1.5,2.1]),key=lambda v:v.cost); a=basis(fit.x); w=ls(a[i],y[i]); print('y',k,name,np.round(score(i,j,a),6),('p '+str(np.round(fit.x,4))) if k==0 else '')
  if k==0 and name=='rat':
   ym=mat(w[5:15]); p=fit.x[1:5]; p/=np.linalg.norm(p); diff=(ym-m2*(np.sum(ym*m2)/np.sum(m2*m2))); print('rat quadratic eig',np.linalg.eigvalsh(ym),'eig-diff',np.linalg.eigvalsh(diff),'linear',w[:5],'quad residual matrix',diff,'error-pred cov',np.corrcoef((a[i]@w-y[i]),T[i].T)[0,1:]); print('state coeff sv',np.linalg.svd(ls(np.c_[U,np.einsum('ni,ij,nj->n',z,m2,z)][i],T[i])[1:],compute_uv=False))
