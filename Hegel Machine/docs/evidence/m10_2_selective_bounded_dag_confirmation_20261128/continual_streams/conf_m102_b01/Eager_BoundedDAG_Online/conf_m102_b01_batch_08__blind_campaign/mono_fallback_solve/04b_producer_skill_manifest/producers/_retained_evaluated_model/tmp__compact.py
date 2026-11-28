import numpy as np,pickle,itertools
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=np.load('/tmp/operators.npz'); z=d['Z']; C=d['C1']; Q=d['C23']; Y=d['Y']; T=Y[:,1:]; ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]; np.set_printoptions(precision=4,suppress=True,linewidth=170)
u,v=s['panel_b']['u'],s['panel_b']['v']; p=PolynomialFeatures(3);f=p.fit_transform(u); names=p.get_feature_names_out(); ix=[[i for i,n in enumerate(names) if len(n.split())==1 or n=='1'],list(range(15)),[i for i,n in enumerate(names) if n=='1' or ' ' not in n or set(n.replace('^2','').replace('^3','').split()) in [{'x0','x1'},{'x1','x2'},{'x2','x3'},{'x0','x3'}]]]; ix +=[[i for i,n in enumerate(names) if n=='1' or ' ' not in n or '^' not in n]]
for I in ix:
 a=f[:,I];q=ls(a,v);h=np.sum(a*np.linalg.pinv(a).T,axis=1);print('B',len(I),'fit',np.std(a@q-v,axis=0),'loo',np.mean(abs((a@q-v)/np.maximum(1-h,1e-8)[:,None]),0))
for k in range(4):
 print('B greedy',k); a=f/(np.linalg.norm(f,axis=0)+1e-9); cols=[0]; old=1
 for j in range(12):
  i=min((i for i in range(a.shape[1]) if i not in cols),key=lambda i:np.linalg.norm(a[:,cols+[i]]@ls(a[:,cols+[i]],v[:,k])-v[:,k])); cols.append(i); g=f[:,cols];q=ls(g,v[:,k]);std=np.std(g@q-v[:,k]); print(names[i],round(std,5),end='; ')
 print()
 print('row v',v[:8,k]);
U,S,V=np.linalg.svd(T-T.mean(0),full_matrices=False); print('time factors',V[:3]); print('t regress y',ls(np.c_[np.ones(len(T)),T],Y[:,0])); print('time examples sorted by amplitude'); ii=np.argsort(U[:,0])[::6];print(np.c_[z[ii],Y[ii]])
# Test algebraic head dictionaries with sparsity and multi-horizon comparisons.
R=[]
for pname,x in [('z',z),('a',d['X']),('zC',np.c_[z,C]),('zQ',np.c_[z,Q]),('zCQ',np.c_[z,C,Q]),('CQ',np.c_[C,Q]),('zCB',np.c_[z,C,d['B']])]:
 for deg in range(1,5 if x.shape[1]==4 else 3):
  for link,g in [('id',x),('tanh',np.tanh(x)),('sin',np.sin(x))]:
   a=PolynomialFeatures(deg).fit_transform(g); a/=np.maximum(1e-7,np.linalg.norm(a[:52],axis=0)); S0,S1,S2=np.linalg.svd(a[:52],full_matrices=False)
   for l in (1e-5,.0001,.001,.01,.03,.1):
    yh=(a[52:72]@S2.T*(S1/(S1*S1+l)))@(S0.T@Y[:52]); e=np.mean(abs(yh-Y[52:]),axis=0); R.append((e[0],e[1:].mean(),pname,deg,link,l))
for j in [0,1]:print('head dictionary',sorted(R,key=lambda x:x[j])[:5])
