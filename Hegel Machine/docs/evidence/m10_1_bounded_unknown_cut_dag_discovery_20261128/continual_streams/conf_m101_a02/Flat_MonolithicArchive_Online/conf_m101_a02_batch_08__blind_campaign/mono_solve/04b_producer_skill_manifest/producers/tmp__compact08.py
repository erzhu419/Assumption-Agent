import sys,pickle,numpy as np
from itertools import combinations
from scipy.linalg import solve
sys.path.insert(0,'/holdout_task')
import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['y']); y=tr['y']; X=tr['x']; z,b,c,d,a,opp=[q[:n] for q in m._blocks(s)]; p=z*np.roll(b,1,axis=1); base=np.c_[m._aff(b),a,opp]
rg=np.random.default_rng(808); folds=[(np.arange(i),np.arange(i,n)) for i in [52,56]]+[(np.setdiff1d(np.arange(n),i),i) for i in np.array_split(rg.permutation(n),4)]
for it,iv in folds:
 t={k:(v[it] if k!='summary' else np.zeros(4)) for k,v in tr.items()}; o=m.solve_batch(dict(s,train=t,test={'x':X[iv]})); print('saved',len(it),np.mean(abs(y[iv]-o['y_hat'])),np.mean(abs(tr['traj'][iv]-o['state_pred'])))
D={'dx2':np.sum((d*X)**2,1),'dd2':np.sum((d*np.roll(d,2,axis=1))**2,1),'dd':np.sum(d*d,1),'bd1s2':np.sum(b*np.roll(d,1,axis=1),1)**2,'cp3':np.sum(c*np.roll(p,3,axis=1),1),'bp':np.sum(b*p,1),'b3e':b[:,3]*np.exp(-1.7*b[:,3]**2),'z2c':z[:,2]**3,'c1s':np.sin(1.7*c[:,1]),'de0':d[:,0]*np.exp(-1.7*d[:,0]**2),'cross24':b[:,0]*b[:,2]-b[:,1]*b[:,3]}
for tag,g in [('d',d),('c',c),('z',z),('b',b)]:
 for k in [0,1,2]:
  u=g*np.roll(g,k,axis=1)
  for q in [.1,.2,.5,1.7]:
   D[f'{tag}sq{k}r{q}']=np.sum(u*u/(1+q*u*u),axis=1)
   D[f'{tag}a{k}r{q}']=np.sum(u/(1+q*u*u),axis=1)
for tag,g in [('z',z),('b',b),('d',d),('c',c)]:
 for norm,h in [('norm',np.sum(g*g,1)),('opp',np.sum(g*np.roll(g,2,axis=1),1)),('first',g[:,0]),('last',g[:,-1])]:
  for q in [.1,.2,.5,1.7]:
   den=1+q*h*h
   for label,w in [('a',a),('opp',opp)]: D[tag+norm+label+str(q)]=w/den

def cv(f):
 return np.array([np.mean(abs(y[j]-f[j]@m._fit(f[i],y[i]))) for i,j in folds])

def rank(e):return np.mean(e[2:])+.3*max(e[2:])
for main in [base,np.c_[base,z],np.c_[base,c],np.c_[base,d]]:
 tests=[(rank(e:=cv(np.c_[main,v])),k,e) for k,v in D.items()]; top=sorted(tests)[:14]; print('main',main.shape,cv(main));print('top single',[(key,np.round(e,4)) for _,key,e in top[:5]])
 keys=list(dict.fromkeys([k for _,k,_ in top]+list(D)[:11])); rr=[]
 for k in [2,3,4]:
  for ks in combinations(keys,k):
   e=cv(np.c_[main,*[D[j] for j in ks]]); rr.append((rank(e),ks,e))
 print('top pooled',[(key,np.round(e,4)) for _,key,e in sorted(rr)[:7]])
 print('passing',[(key,np.round(e,4)) for _,key,e in sorted(r for r in rr if max(r[2][:2])<=.038)[:7]])
