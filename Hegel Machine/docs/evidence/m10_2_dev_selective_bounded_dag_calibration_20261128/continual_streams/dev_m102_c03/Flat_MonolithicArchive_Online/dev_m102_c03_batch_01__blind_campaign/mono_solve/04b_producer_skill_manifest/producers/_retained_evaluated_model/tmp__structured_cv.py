import sys,pickle,numpy as np
from itertools import combinations
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor,cho_solve
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import PolynomialFeatures
sys.path.insert(0,'/holdout_task'); from batch_solver import calibrate,temporal,solve_batch
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; o,g=calibrate(s); z=(tr['x']-o)/g; y=tr['y']; Y=tr['traj']; t=s['eval_times']; n=len(z)
cyc=np.array([(0,1),(0,3),(1,2),(2,3)]); C=z[:,cyc[:,0]]*z[:,cyc[:,1]]; P=PolynomialFeatures(2).fit_transform(z); Q=P@P.T; L=np.c_[np.ones(n),z]; S=z+.55*z*z/np.sqrt(1+.39*z*z); A=np.c_[L,C,S]; B=A@A.T
mat=lambda d: (1+d+d*d/3)*np.exp(-d)
D=cdist(z,z); M=mat(D*np.sqrt(5)/4); E=(C@C.T)*mat(D*np.sqrt(5)/2); J=(C@C.T)*(L@L.T)**2
U=sum(mat(abs(z[:,i,None]-z[None,:,i])*np.sqrt(5)/1.2) for i in range(4)); V=sum(mat(cdist(z[:,[i,j]],z[:,[i,j]])*np.sqrt(5)/1.6) for i,j in combinations(range(4),2)); Vc=sum(mat(cdist(z[:,[i,j]],z[:,[i,j]])*np.sqrt(5)/1.6) for i,j in cyc)
R={}; kernels={}
for nm,K in [('iso',M),('poly',Q+.03*(z@z.T)**3+.003*(z@z.T)**4),('cycle',Q+E),('cyclepoly',Q+.15*J),('additive',Q+U+V),('cycleadd',B+U+Vc+E),('blend',Q+U+V+E+2*M),('base',B+10*M),('low',B+E+U),('cyclesharp',Q+5*E+.2*U),('cyclelo',Q+.2*E+.1*U),('anova3',Q+.5*U+.3*V+.3*(z@z.T)*V+.5*E)]:
 for reg in (1e-6,3e-5,3e-4,.002): kernels[nm,reg]=K+reg*np.eye(n)
splits=[(np.arange(52),np.arange(52,72)),*ShuffleSplit(12,train_size=52,random_state=11).split(z)]
for a,b in splits:
 q,_=temporal(t,Y[a]); T=np.c_[y,Y]; preds=[]; lo=[]
 for name,K in kernels.items():
  L=cho_factor(K[np.ix_(a,a)]); w=cho_solve(L,T[a]); test=K[np.ix_(b,a)]@w; loo=w/np.diag(cho_solve(L,np.eye(len(a))))[:,None]; preds.append(test); lo.append(np.r_[np.mean(abs(loo[:,0])),np.mean(abs(loo[:,1:]))]); R.setdefault(name,[]).append(np.r_[np.mean(abs(test[:,0]-y[b])),np.mean(abs(test[:,1:]-Y[b]))]);
 preds=np.array(preds); lo=np.array(lo); out=[]
 for j in range(2):
  wt=np.exp(-6*(lo[:,j]/lo[:,j].min()-1)); wt/=wt.sum(); pp=np.einsum('k,kij->ij',wt,preds); out.append(np.mean(abs((pp[:,0]-y[b]) if j==0 else (pp[:,1:]-Y[b]))));
 R.setdefault(('adaptive',0),[]).append(out)
 st=dict(s,train={k:(v[a] if k!='summary' else v) for k,v in tr.items()},test={'x':tr['x'][b]}); out=solve_batch(st); R.setdefault(('current',0),[]).append([np.mean(abs(out['y_hat']-y[b])),np.mean(abs(out['state_pred']-Y[b]))]);
for j in range(2):
 print('metric',j)
 for k,a in sorted(R.items(),key=lambda it:np.mean(it[1],0)[j])[:13]+[(k,R[k]) for k in [('adaptive',0),('current',0)]]: print(k,'fixed/mean/max',np.round([a[0],np.mean(a,0),np.max(a,0)],5))
f=PolynomialFeatures(4); X=f.fit_transform(z); labs=f.get_feature_names_out(); q,_=temporal(t,Y); T=np.c_[y,np.linalg.lstsq(q,Y.T,rcond=None)[0].T];
for j in range(3):
 w=np.linalg.solve(X.T@X+np.diag(.005*3.**(f.powers_.sum(1)-2.)),X.T@T[:,j]); print('terms',j,[(labs[i],round(w[i],3)) for i in np.argsort(-abs(w))[:28]])
