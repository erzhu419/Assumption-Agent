import pickle,runpy,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import minimize_scalar, least_squares
from sklearn.model_selection import KFold
p='/holdout_task/'; s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; m=runpy.run_path(p+'node_00.py'); z=m['_calibrate'](s['panel_a'],s['train']['x']); Y=np.c_[s['train']['y'],s['train']['traj']]; print('n, y bounds, trajectory spectrum:',len(z),Y[:,0].min(),Y[:,0].max(),np.linalg.svd(Y[:,1:],compute_uv=False)); print('panel keys:',[(k,list(v)) for k,v in s.items() if k.startswith('panel')]);
fns={'square':lambda x,a:x*x,'cubic':lambda x,a:x**3,'tanh':lambda x,a:np.tanh(a*x),'atan':lambda x,a:np.arctan(a*x),'log':lambda x,a:np.log1p(a*x*x),'sin':lambda x,a:np.sin(a*x),'rat':lambda x,a:x/(1+a*x*x),'soft':lambda x,a:x/np.sqrt(1+a*x*x),'exp':lambda x,a:np.expm1(a*x),'asinh':lambda x,a:np.arcsinh(a*x)}
def fit(u,y):
 n,d=u.shape; scores=[]
 def ev(F):
  q=np.linalg.qr(F,mode='reduced')[0]; r=y-q@(q.T@y); return (np.mean((r/(1-(q*q).sum(1))[:,None])**2)**.5,np.mean(r*r)**.5)
 candidates={'linear':lambda w:np.c_[np.ones(len(w)),w],'cross':lambda w:np.column_stack([np.ones(len(w)),w]+[w[:,i]*w[:,j] for i in range(d) for j in range(i+1,d)]),'quad':lambda w:np.column_stack([np.ones(len(w)),w]+[w[:,i]*w[:,j] for i,j in combinations_with_replacement(range(d),2)])}
 for k,F in candidates.items():scores.append((*ev(F(u)),k,F))
 for name,f in fns.items():
  for linear in [False,True]:
   def design(w,a):return np.c_[np.ones(len(w)),w if linear else np.empty((len(w),0)),f(w,a)]
   grid=np.geomspace(.05,6,40); i=np.argmin([ev(design(u,a))[0] for a in grid]); a=minimize_scalar(lambda a:ev(design(u,a))[0],bounds=(grid[max(i-1,0)],grid[min(i+1,len(grid)-1)]),method='bounded').x
   F=lambda w,a=a,f=f,linear=linear:np.c_[np.ones(len(w)),w if linear else np.empty((len(w),0)),f(w,a)]
   scores.append((*ev(F(u)),f'{name}+linear={linear}@{a:.4f}',F))
 scores.sort(key=lambda q:q[0]); print([(round(a,7),round(b,7),name) for a,b,name,_ in scores[:5]]); F=scores[0][-1]; return F(z)@np.linalg.lstsq(F(u),y,rcond=None)[0]
feat={'z':z}
for panel in ['panel_b','panel_c']:
 for k,y in s[panel].items():
  if k!='u': print(panel,k); feat[k]=fit(s[panel]['u'],y)
np.savez('/tmp/readout_arrays.npz',Y=Y,**feat)
V=np.linalg.svd(Y[:,1:],full_matrices=False)[2][:2]; T=np.c_[Y[:,0],Y[:,1:]@V.T]; print('trajectory rank2 mae',np.abs(T[:,1:]@V-Y[:,1:]).mean());
folds=list(KFold(5,shuffle=True,random_state=31).split(z));
def test(F,T=T,inv=None,reg=.003):
 errs=[]
 for a,b in folds:
  A=F[a]; P=F[b]@np.linalg.solve(A.T@A+np.eye(F.shape[1])*reg,A.T@T[a]);
  if inv is not None:P[:,0]=inv(P[:,0]);
  errs.append(np.abs(np.c_[P[:,0],P[:,1:]@V]-Y[b]).mean(0))
 errs=np.array(errs); return np.round([errs[:,0].mean(),errs[:,1:].mean(),errs[:,0].max(),errs[:,1:].mean(1).max()],5)
for tags in [('z',),('v',),('v1',),('v2',),('v1','v2'),('v','v1','v2'),('z','v','v1','v2')]:
 X=np.column_stack([feat[k] for k in tags]); X=(X-X.mean(0))/X.std(0); A=np.c_[np.ones(len(X)),X]; Q=np.column_stack([A]+[X[:,i]*X[:,j] for i,j in combinations_with_replacement(range(X.shape[1]),2)]); print(tags,'linear',test(A),'quad',test(Q,reg=.05),'atanh',[(L,test(A,np.c_[np.arctanh(Y[:,0]/L),T[:,1:]],lambda q:L*np.tanh(q))[0]) for L in [1.2,1.3,1.5,2.] if L>np.max(np.abs(Y[:,0]))])
