import sys,pickle,numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import ShuffleSplit
sys.path.insert(0,'/holdout_task'); from batch_solver import calibrate,temporal
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; off,g=calibrate(s); z=(tr['x']-off)/g; t=s['eval_times']; y=tr['y']; Y=tr['traj']; z=z/.9; p=PolynomialFeatures(6); F=p.fit_transform(z); deg=p.powers_.sum(1); supp=(p.powers_>0).sum(1); nm=p.get_feature_names_out(); G=[]
for i in range(7):
 for j in range(1 if i else 0,min(i,4)+1):
  v=F[:,(deg==i)&(supp==j)]; G.append(v@v.T)
G=np.array(G); splits=[(np.arange(52),np.arange(52,72)),*ShuffleSplit(8,train_size=52,random_state=11).split(z)]; R=[]
for ii,(a,b) in enumerate(splits):
 q,A=temporal(t,Y[a]); T=np.c_[y,np.linalg.lstsq(q,Y.T,rcond=None)[0].T]; P=[]
 for jj,j in enumerate([0,1,2]):
  Ta=T[a,j]; k=G[:,a[:,None],a]; kb=G[:,b[:,None],a]; k=np.concatenate([k,np.eye(len(a))[None]],0); init=np.r_[np.arange(len(G))*-.5,-11.]; init=np.maximum(init,-11.)
  def fun(log):
   w=np.exp(log); K=np.einsum('i,ijk->jk',w,k); L=cho_factor(K); v=cho_solve(L,Ta); Ki=cho_solve(L,np.eye(len(a))); f=.5*Ta@v+np.log(L[0].diagonal()).sum(); d=.5*np.einsum('ijk,jk->i',k,Ki-np.outer(v,v))*w
   return f,d
  opt=minimize(fun,init,jac=True,method='L-BFGS-B',bounds=[(-24,3)]*len(G)+[(-14,-4)],options=dict(maxiter=160)); w=np.exp(opt.x); pred=np.einsum('i,ijk->jk',w[:-1],kb)@np.linalg.solve(np.einsum('i,ijk->jk',w,k),Ta); P.append(pred)
  if ii==0:
   for d in [3,4]:
    S=F[:,deg<=d]; c=np.linalg.solve(S[a].T@S[a]+.01*np.eye(S.shape[1]),S[a].T@Ta); print('coefs',j,d,[(nm[i],round(c[i],3)) for i in np.argsort(-abs(c))[:16]])
 P=np.array(P).T; ey=np.mean(abs(P[:,0]-y[b])); et=np.mean(abs(P[:,1:]@q.T-Y[b])); R.append((ey,et));print(ii,ey,et)
print('mean/max',np.mean(R,0),np.max(R,0))
