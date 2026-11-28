import pickle,sys,itertools,numpy as np
from calibration_trial import _fit_panel_a,_forward_a,_inverse_a
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; a=s['panel_a']; b=s['panel_b']; c=_fit_panel_a(a['u'],a['v']); M=np.linalg.lstsq(np.c_[np.ones(len(b['u'])),b['u']],b['v'],rcond=None)[0]; x=tr['x']; ya=tr['y']; ta=tr['traj']
B=lambda z:z@M[1:]+M[0]; Bi=lambda z:np.linalg.solve(M[1:].T,(z-M[0]).T).T
ops={'A+':lambda z:_forward_a(z,c),'A-':lambda z:_inverse_a(z,c),'B+':B,'B-':Bi,'B0+':lambda z:z@M[1:],'B0-':lambda z:np.linalg.solve(M[1:].T,z.T).T}

def mae(X,y):
 e=[]
 for te in np.array_split(np.arange(len(X)),6):
  ti=np.setdiff1d(np.arange(len(X)),te); e.append(np.mean(abs(X[te]@np.linalg.lstsq(X[ti],y[ti],rcond=None)[0]-y[te])))
 return max(e)

def score(z):
 h=z[:,0]*z[:,2]+z[:,1]*z[:,3]
 return [mae(np.c_[z,h],ya),mae(z,ta),mae(np.c_[np.ones(len(z)),z],ta),mae(np.c_[np.ones(len(z)),z,h],ya)]
results=[]
for l in range(4):
 for seq in itertools.product(ops,repeat=l):
  if l>1 and any(p[0]==q[0] and p[-1]!=q[-1] for p,q in zip(seq,seq[1:])): continue
  z=x
  try:
   with np.errstate(all='raise'):
    for op in seq:z=ops[op](z)
    scores=score(z)
  except (FloatingPointError,np.linalg.LinAlgError): continue
  results.append((seq,np.round(scores,7).tolist()))
for j in range(4):
 print('RANK',j); print(*sorted(results,key=lambda row:row[1][j])[:8],sep='\n')
print('means x / A- / B+ / B- / A calibration u / B calibration u');print(np.stack([q.mean(0) for q in (x,ops['A-'](x),B(x),Bi(x),a['u'],b['u'])]));print('trajectory SVD',np.linalg.svd(ta,compute_uv=False))
