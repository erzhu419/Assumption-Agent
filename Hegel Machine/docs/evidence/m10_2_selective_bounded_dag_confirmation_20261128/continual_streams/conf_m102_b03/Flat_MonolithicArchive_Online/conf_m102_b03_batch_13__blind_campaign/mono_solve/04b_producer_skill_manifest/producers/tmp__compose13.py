from search13 import *
x=tr['x']; z,B,P,Q=setup(s,x); p,q=P(z),Q(z); b=D(z)@B; Y=np.c_[tr['y'],tr['traj']]; t=s['eval_times']
def score(A,Y=Y):
 I=np.linalg.pinv(A[:52],rcond=1e-8); w=I@Y[:52]; r=(A[:52]@w-Y[:52])/(1-np.sum(A[:52]*I.T,axis=1))[:,None]; e=A[52:]@w-Y[52:];return np.mean(abs(r),axis=0),np.mean(abs(e),axis=0),w
adj=lambda a:a*np.roll(a,-1,axis=1)
for name,a in [('pq',np.c_[p,q]),('pqmul',np.c_[p,q,p*q]),('pqmulz',np.c_[p,q,p*q,z]),('pqmulrat',np.c_[p,q,p*q,z/(1+.22*z*z)]),('pqmulsoft',np.c_[p,q,p*q,z/(1+abs(z))]),('pqmulcub',np.c_[p,q,p*q,z,z**3]),('pqsummul',np.c_[p,q,np.sum(p*q,axis=1)]),('pqallmul',np.c_[p,q,(p[:,:,None]*q[:,None,:]).reshape(len(z),16)]),('pqmulzadj',np.c_[p,q,p*q,z,adj(z)]),('pqsumz',np.c_[p,q,np.sum(p*q,axis=1),z]),('pqmul2',np.c_[p,q,p*q,p**2,q**2]),('pqmulroll',np.c_[p,q,p*q,p*np.roll(q,-1,axis=1)])]:
 cv,e,w=score(D(a)); print(name,len(w), 'y/trajectory/terminal', np.round([cv[0],cv[1:].mean(),cv[-1],e[0],e[1:].mean(),e[-1]],5));
 if name in ('pqmul','pqmulz'):print(np.round(w[:,[0,1,-1]].T,4))
G=groups(s,x);G['core']=np.c_[p,q,p*q];G['corez']=np.c_[p,q,p*q,z];G['coreSum']=np.c_[p,q,np.sum(p*q,axis=1)];G['coreSumz']=np.c_[G['coreSum'],z];
for obj,outs in [('scalar',Y[:,:1]),('trajectory',Y[:,1:])]:
 for base in ['core','corez','coreSum','coreSumz']:
  A=D(G[base]); results=[]
  for nm,F in G.items():
   C=np.c_[A,F];cv,e,w=score(C,outs);results.append((cv.mean(),e.mean(),nm))
  print(obj,base,sorted(results)[:12]);
print('trajectory residual spectrum')
for rates in [[2.15,.22],[2.15,.45],[1.8041,.4062],[2.15,.22,1.],[2.15,.22,4.]]:
 H=-np.expm1(-t[:,None]*rates);E=tr['traj'][:52].T-H@L(H,tr['traj'][:52].T); print(rates,np.sqrt(np.mean(E**2)),np.linalg.svd(E,compute_uv=False)[:4]);
