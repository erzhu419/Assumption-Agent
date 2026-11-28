import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']; X=D['true_channel_loads']; Y=D['bridge_raw']; clean=np.max(abs(Y),axis=1)<5
pairs=list(combinations_with_replacement(range(4),2)); triples=list(combinations_with_replacement(range(4),3))
def design(x,j,k,p,degree):
 z=x[:,j]; A=[np.ones(len(x)),*x.T]
 if degree>=2:A.extend(x[:,a]*x[:,b] for a,b in pairs)
 if degree>=3:A.extend(x[:,a]*x[:,b]*x[:,c] for a,b,c in triples)
 if k=='poly5': A.extend([z**4,z**5])
 if k in ('tanh','atan','softsign','sine','rational'):
  a,b=p; u=a*(z-b)
  g={'tanh':np.tanh(u),'atan':np.arctan(u),'softsign':u/np.sqrt(1+u*u),'sine':np.sin(u),'rational':u/(1+u*u)}[k]; A.append(g)
 return np.column_stack(A)
def fit(t,j,k,degree):
 def predict(p,v):
  A=design(X[t],j,k,p,degree); beta=np.linalg.lstsq(A,Y[t,j],rcond=None)[0];return design(X[v],j,k,p,degree)@beta,beta
 p=()
 if k in ('tanh','atan','softsign','sine','rational'):
  r=min((least_squares(lambda p:predict(p,t)[0]-Y[t,j], [a,0.],bounds=([.08,-1.],[5.,1.]),max_nfev=90) for a in (.55,1.4,2.8)),key=lambda r:r.cost);p=r.x
 return lambda v:predict(p,v),p
axis=np.flatnonzero(clean & (np.count_nonzero(X,axis=1)==1)); mixed=np.flatnonzero(clean & (np.count_nonzero(X,axis=1)!=1)); splits=[('mix',v) for v in np.array_split(np.random.default_rng(49).permutation(mixed),4)]+[('axis',axis),('extreme',mixed[np.max(abs(X[mixed]),axis=1)>1.48])]
for k,degree in [('affine',1),('poly',2),('poly3',3),('poly5',3),('tanh',1),('tanh',2),('atan',2),('softsign',2),('rational',2),('sine',2),('tanh',3),('atan',3)]:
 err={}; params=[]; coefs=[]; full=[]
 for j in range(4):
  pred,p=fit(clean,j,k,degree); yh,beta=pred(clean); params.append(np.round(p,4).tolist()); coefs.append(beta); full.append(np.sqrt(np.mean((yh-Y[clean,j])**2)))
  for group,v in splits:
   t=clean.copy();t[v]=False; pred,p=fit(t,j,k,degree);r=pred(v)[0]-Y[v,j];e=err.setdefault(group,np.zeros(4)); e[j]=max(e[j],np.sqrt(np.mean(r*r)))
 print(k,degree,'fit',np.round(full,6),'CV',{k:np.round(v,6).tolist() for k,v in err.items()},'params',params,flush=True)
 if k=='tanh' and degree==2: print('coefs',np.round(np.array(coefs).T,5).tolist(),flush=True)
