import sys,pickle,copy
import numpy as np
from scipy.linalg import solve
sys.path.insert(0,'/holdout_task/_frozen_nodes')
import node_00,node_02
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; n=len(s['train']['y'])
def batches():
 rng=np.random.default_rng(51)
 splits=[np.arange(52,n),*np.array_split(rng.permutation(n),5)]
 splits += [np.argsort(s['train']['x'][:,j])[-14:] for j in range(4)]
 splits += [np.argsort(np.linalg.norm(s['train']['x'],axis=1))[-14:]]
 for I in splits:
  J=np.setdiff1d(np.arange(n),I); t=copy.deepcopy(s); t['train']={k:v[J] if len(v)==n else v for k,v in s['train'].items()}; t['test']['x']=s['train']['x'][I]
  p=node_00.run_node({'__input__':t}); pred=node_02.run_node({'node_00':p})['state_pred']; yield p,pred,s['train']['y'][I],s['train']['traj'][I]
def read(p,pred,kind,ridge):
 A,B=p['scores'],pred@p['modes'].T
 maps={'s':(p['sy'],p['qy']),'t':(p['st'],p['qt']),'u':(p['st'][:,:4],p['qt'][:,:4]),'a':(A,B),'b':(node_00._quadratic(A),node_00._quadratic(B))}
 K=np.zeros_like(p['K']); L=np.zeros_like(p['Kq'])
 for f in kind:
  if f=='k': K+=p['K']; L+=p['Kq']; continue
  U,V=maps[f]; K+=U@U.T;L+=V@U.T
 # profile an intercept via centering the Gram matrix
 c=K.mean(axis=0); L=L-c-L.mean(axis=1,keepdims=True)+c.mean(); K=K-c-c[:,None]+c.mean()
 return p['y_mean']+L@solve(K+ridge*np.eye(len(K)),p['y_centered'],assume_a='pos')
if __name__=='__main__':
 D=list(batches()); print('trajectory',np.round([np.abs(pr-tr).mean() for _,pr,_,tr in D],5))
 for kind in ['k','s','t','a','b','ks','ka','ksa','kst','kstb','ksta','sta','stb','stu','sua','ta','tb','ktb','kta','kt','ksua','ksub','sa','sb','st']:
  vals=[]
  for r in [1e-7,1e-5,.0003,.003,.03,.3,3]:
   e=np.array([np.abs(read(p,pr,kind,r)-y).mean() for p,pr,y,_ in D]); vals.append((max(e),r,e))
  w,r,e=min(vals,key=lambda z:z[0]); print(kind,r,np.round(e,5),'worst',round(w,5),'mean',round(e.mean(),5),'public range',np.round([v[2][0] for v in vals],5))
