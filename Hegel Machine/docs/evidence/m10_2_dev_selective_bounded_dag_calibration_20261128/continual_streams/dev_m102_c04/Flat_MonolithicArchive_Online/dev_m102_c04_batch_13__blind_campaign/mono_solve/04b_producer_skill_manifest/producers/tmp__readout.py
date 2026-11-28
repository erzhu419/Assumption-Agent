import itertools as it, pickle
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import qr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; q=np.load('/tmp/identified.npz'); Y=q['Y']; p,r=q['p'],q['r']; H=np.c_[p,r]; t=s['eval_times']; n=52
np.set_printoptions(precision=5,suppress=True,linewidth=140)
def ls(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def R(k):return -np.expm1(-t[:,None]*k)
def terr(k):
 A=R(k); return (A@ls(A,Y[:n,1:].T)-Y[:n,1:].T).ravel()
k=least_squares(terr,[1.8,.43],bounds=([.4,.001],[10,1.5])).x
A=R(k); print('rates',k,'time RMSE',np.sqrt(np.mean(terr(k)**2))); Z=np.c_[Y[:,0],ls(A,Y[:,1:].T).T]
terms=list(it.combinations_with_replacement(range(8),2)); V=np.c_[np.ones(len(H)),H,np.array([H[:,i]*H[:,j] for i,j in terms]).T]; labels=['1']+[f'{a}{i}' for a in 'pr' for i in range(4)]+[f'{"pr"[i//4]}{i%4}*{"pr"[j//4]}{j%4}' for i,j in terms]
for j,(name,w,w52) in enumerate(zip(['y','fast','slow'],ls(V[:72],Z).T,ls(V[:52],Z[:52]).T)):
 print(name,'coefs (all,52)')
 for i in np.argsort(-np.abs(w))[:26]:print(labels[i],w[i],w52[i])
 print('in RMSE',np.sqrt(np.mean((V[:72]@w-Z[:,j])**2)))
for keys in (('p','r','pp','rr','pr'),('p','r','pr'),('p','r','pp','pr'),('p','r','rr','pr'),('p','r','dot'),('p','r','diag'),('p','r','diag','pp','rr')):
 cols=[V[:,0]]
 for key in keys:
  if len(key)==1:cols.append(p if key=='p' else r)
  elif key=='dot':cols.append(np.sum(p*r,axis=1))
  elif key=='diag':cols.append(p*r)
  else:cols.extend(V[:,i] for i,l in enumerate(labels) if '*' in l and l[0]+l[3]==key)
 X=np.column_stack(cols); zz=X[52:72]@ls(X[:52],Z[:52]);print(keys,'mae',np.mean(abs(zz[:,0]-Z[52:,0])),np.mean(abs(zz[:,1:]@A.T-Y[52:,1:])), 'fit rmse',np.sqrt(np.mean((X[:52]@ls(X[:52],Z[:52])-Z[:52])**2,axis=0)))
for j,key in enumerate(('v1','v2')):
 u=s['panel_c']['u']; v=s['panel_c'][key]
 def F(x,k):return np.c_[np.ones(len(x)),x,x/(1+k*abs(x))] if j==0 else np.c_[np.ones(len(x)),x,np.arctan(k*x),np.array([x[:,a]*x[:,b] for a,b in it.combinations(range(4),2)]).T]
 k=[.8361188575,1.0725818863][j]; f=F(u,k); print(key,'panel coeffs',ls(f,v));print('panel residual',np.sqrt(np.mean((v-f@ls(f,v))**2)))
