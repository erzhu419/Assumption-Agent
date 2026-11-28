import pickle,runpy,numpy as np
from scipy.linalg import cho_factor,cho_solve
D='/holdout_task/'
s=pickle.load(open(D+'public_batch.pkl','rb'))[0]['study']; d=runpy.run_path(D+'_frozen_nodes/node_00.py')['run_node']({'__input__':s}); node=runpy.run_path(D+'node_01.py')['run_node']; y=d['y']; T=d['traj']; z=d['z']; ids=np.arange(len(y))
folds=[ids[52:],*np.array_split(np.random.default_rng(92).permutation(len(y)),5)]
# Candidate and comparison banks use only the fitting fold for scaling and weighting.
views=('i2b','x2b','a2','ij','x','ib')
def bank(i,j):
  x=z['x'][np.r_[i,j]]; x=(x-x[:len(i)].mean(0))/x[:len(i)].std(0); G=(1+x@x.T/4)**2
  n=len(i); I=np.eye(n); yc=y[i]-y[i].mean(); ans=[]
  for k in views:
    w=z[k][np.r_[i,j]]; w=(w-w[:n].mean(0))/w[:n].std(0); d=(w[:,None,:]-w[None,:,:])**2
    for l in (1.1,1.5,2.2,3.2):
      r=d.sum(2)/l**2; R=np.exp(-r/2); h=np.sqrt(5*r); M=(1+h+h*h/3)*np.exp(-h); A=np.exp(-d/(2*l*l)).mean(2)**2
      for C in (R,R*.5+A*.5,M):
        for a in (.3,1,2):
          K=G+a*C
          for e in (.0015,.008,.02):
            L=cho_factor(K[:n,:n]+e*I); b=cho_solve(L,yc); diag=cho_solve(L,I).diagonal(); loo=np.mean(abs(b/diag)); ml=np.log(yc@b/n)+2*np.log(np.diag(L[0])).sum()/n
            ans.append((loo,ml,y[i].mean()+K[n:,:n]@b))
  return ans
rows=[]
for j in folds:
  i=np.setdiff1d(ids,j); zz={k:v[np.r_[i,j]] for k,v in z.items()}; out=node({'node_00':{'z':zz,'y':y[i],'traj':T[i]}})
  ans=bank(i,j); scores=np.array([(q[0],q[1]) for q in ans]); P=np.array([q[2] for q in ans]); pred=[out['y_hat']]
  for sc,temp in zip(scores.T,(.006,.07)):
    w=np.exp(-(sc-sc.min())/temp); pred.append(w@P/w.sum())
  pred += [(pred[1]+pred[2])/2,(pred[0]+pred[1])/2]
  rows.append([*[np.mean(abs(p-y[j])) for p in pred],np.mean(abs(out['state_pred']-T[j])),np.max(np.mean(abs(out['state_pred']-T[j]),axis=0))]);
print('columns: current, LOO ensemble, ML ensemble, adaptive blend, current+LOO, trajectory, worst trajectory horizon'); print(np.round(rows,6)); print('worst',np.round(np.max(rows,axis=0),6))
