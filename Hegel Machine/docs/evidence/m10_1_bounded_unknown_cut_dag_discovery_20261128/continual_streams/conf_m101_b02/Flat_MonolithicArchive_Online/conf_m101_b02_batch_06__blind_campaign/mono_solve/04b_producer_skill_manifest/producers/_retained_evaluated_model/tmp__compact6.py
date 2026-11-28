from readout6 import *
x=s['train']['x']; a,b,c,d=nodes(x); n=len(a); Y=s['train']['y']; T=s['train']['traj']; t=s['eval_times']
p=a*c; q=a*np.roll(c,1,axis=1)
G={'full':D(a,c,(a[:,:,None]*c[:,None,:]).reshape(n,16)), 'cyclic':D(a,c,p,q), 'cyclic1':D(a,c,p,q.sum(1)), 'cyclic2':D(a,c,p.sum(1),q.sum(1)), 'cyclic3':D(a,c,p.sum(1),q), 'cyclic4':D(a,c,(p+1.65*q).sum(1))}
def modes(r): return -np.expm1(-t[:,None]*r)
def tm(T, k):
 def res(r):
  M=modes(r); return (M@fit(M,T.T)-T.T).ravel()
 q0=[.4,1.8,5.2][:k]; low=[.1,.9,3][:k]; high=[.9,3,10][:k]
 return modes(least_squares(res,q0,bounds=(low,high),gtol=1e-10).x)
folds=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72))]+[(np.r_[0:i,i+18:72],np.arange(i,i+18)) for i in (0,18,36,54)]
stats={}
for I,J in folds:
 B={k:tm(T[I],k) for k in [2,3]}
 for key,g in G.items():
  cy=fit(g[I],Y[I]); ct=fit(g[I],T[I]); py=g[J]@cy; pt=g[J]@ct
  e=[np.mean(abs(py-Y[J])),np.mean(abs(pt-T[J]))]
  for k,m in B.items():
   v=pt@ (m@fit(m,np.eye(len(t)))).T; e.append(np.mean(abs(v-T[J])))
  e.extend([np.mean(abs(pt[:,-1]-T[J,-1])),np.mean(abs((pt@ (B[3]@fit(B[3],np.eye(len(t)))).T)[:,-1]-T[J,-1]))]); stats.setdefault(key,[]).append(e)
  if len(stats[key])==1:
   print(key,'fit RMSE',np.std(g[I]@cy-Y[I]),np.std(g[I]@ct-T[I]),'scalar coefs',cy.round(4),'three-mode coefficients',(ct@fit(B[3],np.eye(len(t))).T).round(3) if key in ('cyclic','cyclic2') else '')
for k,v in stats.items(): print(k,'52/20, mean, worst: y/free/two/three/free-final/three-final',np.round(np.stack([v[0],np.mean(v,axis=0),np.max(v,axis=0)]),6).tolist())
