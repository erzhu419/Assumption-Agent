import numpy as np,pickle
from itertools import combinations,product
from scipy.optimize import least_squares
from sklearn.model_selection import KFold
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; x=t['x']; z=t['traj']; V=np.linalg.svd(z,full_matrices=False)[2][:2]; Y=np.c_[t['y'],z@V.T/np.sqrt(z.shape[1])]; A=s['panel_a']; C=s['panel_c']; aff=lambda x:np.c_[np.ones(len(x)),x]; i=np.argsort(np.linalg.norm(A['u']-A['v'],axis=1))[:-3]; M=np.linalg.lstsq(aff(A['u'])[i],A['v'][i],rcond=None)[0]; xc=np.linalg.solve(M[1:].T,(x-M[0]).T).T; L=np.linalg.lstsq(aff(C['u']),C['v1'],rcond=None)[0]; U,_,W=np.linalg.svd(L[1:]); np.set_printoptions(precision=4,suppress=True); print('panel A',M,'target spectrum',np.linalg.svd(Y,compute_uv=False)); print('target covariance',np.corrcoef(Y.T));print('Y least squares against other state',np.mean(abs(Y[:,0]-aff(Y[:,1:])@np.linalg.lstsq(aff(Y[:,1:]),Y[:,0],rcond=None)[0])));
folds=list(KFold(6,shuffle=True,random_state=12).split(x)); out=[]
def err(p,j):
 e=p-Y[j]; return np.array([np.abs(e[:,0]).mean(),np.abs(e[:,1:]@V).mean()*np.sqrt(z.shape[1])])
def score(F,lam=.001):
 E=[]
 for i,j in folds:
  f=F[i]; E.append(err(F[j]@np.linalg.solve(f.T@f+lam*np.eye(f.shape[1]),f.T@Y[i]),j))
 return np.mean(E,0)
for nam,q in [('x',x),('cal',xc),('C',aff(xc)@L),('orth',xc@U@W)]:
 R=np.sum(q*q,axis=1); G=aff(q); Q=np.column_stack([G]+[q[:,a]*q[:,b] for a,b in combinations(range(4),2)]+[q*q]);
 for f in np.arange(.5,7,.15):
  for n,F in [('periodic',np.c_[G,np.sin(f*q),np.cos(f*q)]),('radial',np.c_[Q,np.cos(f*np.sqrt(R)),np.sin(f*R),np.cos(f*R)]),('radial-rational',np.c_[G,Q[:,5:]/(1+f*R[:,None]),G*np.exp(-f*R[:,None])]),('add-rational',np.c_[G,q*q/(1+f*q*q),q/(1+f*q*q)]),('sine-square',np.c_[G,np.sin(f*q*q),np.cos(f*q*q)])]:out.append((score(F),nam,n,round(f,2)))
for j in [0,1]:print('univariate',j, *sorted(out,key=lambda a:a[0][j])[:8],sep='\n')
# Profile one nonlinear ridge atom jointly, testing single-index oscillation, projected square, or rational saturation.
for nam,q in [('raw',x),('cal',xc)]:
 G=aff(q); Q=np.c_[G,q*q,*[q[:,a]*q[:,b] for a,b in combinations(range(4),2)]]
 for base in [G,Q]:
  for kind in ['wave','squarewave','tanh','rational','bump']:
   def form(w,idx):
    a=G[idx]@w
    if kind=='wave': F=np.c_[np.sin(a),np.cos(a)]
    elif kind=='squarewave': F=np.sin(a*a)[:,None]
    elif kind=='tanh': F=np.tanh(a)[:,None]
    elif kind=='rational': F=(a/(1+a*a))[:,None]
    else: F=np.exp(-a*a)[:,None]
    return np.c_[base[idx],F]
   def fit(i):
    def prof(w,ret=False):
     F=form(w,i); B=np.linalg.lstsq(F,Y[i],rcond=1e-8)[0];return B if ret else (F@B-Y[i]).ravel()
    rng=np.random.default_rng(249); starts=[np.r_[0,2.15*d] for d in np.eye(4)]+list(rng.normal(size=(20,5))*1.5); fits=[least_squares(prof,u,max_nfev=110,bounds=(-21,21)) for u in starts]; m=min(fits,key=lambda r:r.cost); return m.x,prof(m.x,True),m.cost
   ii=np.arange(len(q)); w,b,cost=fit(ii); errors=[]
   for i,j in folds:
    wf,bf,_=fit(i);errors.append(err(form(wf,j)@bf,j))
   print(nam,base.shape[1],kind,'train',err(form(w,ii)@b,ii),'CV',np.mean(errors,0),'w',w,flush=True)
