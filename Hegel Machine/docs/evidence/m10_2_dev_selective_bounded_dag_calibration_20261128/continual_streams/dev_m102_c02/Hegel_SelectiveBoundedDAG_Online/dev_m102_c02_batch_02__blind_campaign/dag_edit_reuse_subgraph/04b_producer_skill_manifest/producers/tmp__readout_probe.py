import itertools, pickle, runpy
import numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':s})['c']; y=s['train']['y']; panel=s['panel_c']; n=len(y)
pairs=list(itertools.combinations_with_replacement(range(4),2)); base=np.column_stack((np.ones(n),c)); q=np.array([c[:,i]*c[:,j] for i,j in pairs]).T
print('coord means/ranges',c.mean(0),np.ptp(c,axis=0)); print('terms',pairs)
def fit(X):
 b=np.linalg.lstsq(X[:52],y[:52],rcond=None)[0]; e=abs(X@b-y); return np.mean(e[52:]),np.mean(e[:52]),b
for name,Q in [('linear',q[:,:0]),('archive',(q[:,2]+q[:,6])[:,None]),('full',q)]: print(name,fit(np.c_[base,Q]))
for k in range(1,5):
 scores=[]
 for ix in itertools.combinations(range(10),k):
  a,b,coef=fit(np.c_[base,q[:,ix]]); scores.append((a,b,tuple(pairs[i] for i in ix),coef[5:]))
 print('best',k,*sorted(scores,key=lambda item:item[1])[:4],sep='\n')
for key in ('v1','v2'):
 u=panel['u']; v=panel[key]; X=np.c_[np.ones(len(u)),u]; coef=np.linalg.lstsq(X,v,rcond=None)[0]; print('panel',key,'affine max err',np.max(abs(X@coef-v)), 'coef',coef)
