import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task'); from batch_solver import _branches,_predict
from sklearn.linear_model import lars_path
from scipy.linalg import solve
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; Y=np.c_[tr['y'],tr['traj']]; Q=_branches(s,tr['x']); folds=[(np.setdiff1d(np.arange(52),j),j) for j in np.array_split(np.random.default_rng(624).permutation(52),4)]+[(np.arange(52),np.arange(52,72))]
def dictionary(Q,variant):
 Z=np.column_stack(Q); n,m=Z.shape; i,j=np.triu_indices(m); a,b=Z[:,i],Z[:,j]; nms=[f'x{k}' for k in range(m)]; blocks=[Z,a*b]; names=nms+[f'x{k}*x{l}' for k,l in zip(i,j)];
 if variant>=1:
  for t in [.3,1,3]:
   for name,F in [('rat',lambda v:v/(1+t*v*v)),('tanh',lambda v:np.tanh(t*v)),('cos',lambda v:np.cos(t*v))]:
    blocks += [F(Z),F(a+b),F(a-b)]; names += [f'{name}{t}({v})' for v in nms]+[f'{name}{t}(x{k}+x{l})' for k,l in zip(i,j)]+[f'{name}{t}(x{k}-x{l})' for k,l in zip(i,j)]
   blocks += [a/(1+t*b*b),b/(1+t*a*a)]; names += [f'x{k}/({t}x{l})' for k,l in zip(i,j)]+[f'x{l}/({t}x{k})' for k,l in zip(i,j)]
 if variant>=2:
  if m<24:
   ijk=list(combinations(range(m),3)); blocks +=[np.prod(Z[:,ijk],axis=2)]; names += [str(k) for k in ijk]
  for x in Q:
   blocks +=[x*np.roll(x,1,1)*np.roll(x,2,1)];names+=['cycle3']*x.shape[1]
 return np.column_stack(blocks),np.array(names)
if __name__=='__main__':
 for k,z in enumerate(Q): print('branch',k,'range',z.min(0),z.max(0));
 print('cal cov',np.corrcoef(np.c_[*Q,Y[:,0]].T)[-1].round(2));
 out=[]
 for group,parts in [('base',Q),('noC2',Q[:3]),('C',Q[2:]),('B',Q[1:2]),('x',Q[:1])]:
  for v in range(3):
   D,names=dictionary(parts,v); rec=[]; coeff=[]
   for i,j in folds:
    x=(D-D[i].mean(0))/np.maximum(D[i].std(0),1e-8); err=[]; co=[]
    for y in [Y[:,0],Y[:,-1]]:
     mu=y[i].mean(); _,_,w=lars_path(x[i],y[i]-mu,max_iter=85,method='lasso');
     outp=x[j]@w+mu; e=np.mean(abs(outp-y[j,None]),axis=0); err.append(np.interp(np.arange(1,41),np.count_nonzero(w,axis=0),e)); co.append(w)
    rec.append(err); coeff.append(co)
   a=np.array(rec); ii=a[:4].mean(0).argmin(1); print(group,v,[(int(k+1),round(a[:4,l,k].mean(),5),round(a[-1,l,k],5)) for l,k in enumerate(ii)],flush=True); k=ii[0]; w=coeff[-1][0]; sel=np.argmin(abs(np.count_nonzero(w,axis=0)-k-1)); idx=np.argsort(abs(w[:,sel]))[-7:]; print(list(zip(names[idx],w[idx,sel].round(3))),flush=True)
