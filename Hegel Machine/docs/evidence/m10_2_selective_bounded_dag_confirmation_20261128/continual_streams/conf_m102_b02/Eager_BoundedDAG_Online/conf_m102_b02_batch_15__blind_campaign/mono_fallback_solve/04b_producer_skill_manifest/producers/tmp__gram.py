import sys;sys.path.insert(0,'/holdout_task')
from batch_solver import *
import pickle,itertools as I
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=np.load('/tmp/design.npz'); z,b,c,h=[p[k][:72] for k in 'zbch']; N=52; M=p['m']; y=p['y']; names=[]; G=[]
def add(k,a):
 a=np.asarray(a).reshape(72,-1); names.extend([f'{k}:{i}' for i in range(a.shape[1])]); G.extend(a.T)
for k,A in zip('zbch',(z,b,c,h)):
 add(k,A);add(k+'q',quad(A));add(k+'3',A**3)
 for q in [1.25,2.15,2.6]:
  for m,F in [('s',np.sin), ('c',np.cos), ('t',np.tanh), ('r',lambda u:u/(1+u*u)),('e',lambda u:np.exp(-u*u))]:add(f'{k}{m}{q}',F(q*A))
 for l,B in zip('zbch',(z,b,c,h)):
  if k<l: add(k+l,A[:,:,None]*B[:,None,:])
for i,j,k in I.combinations(range(4),3):add(f'z{i}{j}{k}',z[:,i]*z[:,j]*z[:,k])
for q in [.22,.4,.6,1,1.25,2.15]:
 for k,A in zip('zbch',(z,b,c,h)):
  den=1+q*A*A
  for l,B in zip('zbch',(z,b,c,h)):
   add(f'{l}/{k}{q}',B[:,:,None]/den[:,None,:]);
   if l=='b':add(f'bq/{k}{q}',quad(B)[:,:,None]/den[:,None,:]);
G=np.array(G).T
for j,base in [(0,d(z)),(1,d(quad(b)))]:
 F=base; yy=M[:,j]; print('head',j)
 for i in range(13):
  R=yy-F@fit(F[:N],yy[:N]); Z=G-F@fit(F[:N],G[:N]); norms=np.linalg.norm(Z[:N],axis=0); norms[norms<1e-5]=np.inf; scores=np.abs(Z[:N].T@R[:N])/norms; k=scores.argmax(); print(i,round(np.abs(R[:N]).mean(),4),round(np.abs(R[N:]).mean(),4),names[k]);F=np.column_stack([F,G[:,k]])
for A in [z,b,c,h]: print('coordinate residual corr',np.corrcoef(np.c_[M-d(z,quad(b))@fit(d(z,quad(b))[:N],M[:N]),A].T)[:2,2:])
