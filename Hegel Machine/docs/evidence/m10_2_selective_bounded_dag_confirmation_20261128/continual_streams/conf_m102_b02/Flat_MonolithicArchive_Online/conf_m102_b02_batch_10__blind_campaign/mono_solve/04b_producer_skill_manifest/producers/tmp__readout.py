import numpy as np,pickle
from operators import *
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; z=calibrate(s['panel_a'],tr['x']); bs,c1,c2=operators(s)
u,v=s['panel_b']['u'],s['panel_b']['v']; e,q,c=fit(lambda q:I(np.c_[u,np.sin(q[0]*u)]),v,[[1.25],[2.15],[3]]); print('B sin coeff',q,c.round(4)); print('B best output residuals',np.std(bs[0][1](u)-v,axis=0));
Y=np.c_[tr['y'],tr['traj']]; n=52; combos=[]
for bn,b in bs[:1]+[('sine',lambda a:I(np.c_[a,np.sin(q[0]*a)])@c)]:
 for pref,A in [('z',z),('Bz',b(z))]:
  for suff,D in [('',A),('1',c1(A)),('2',c2(A)),('12',c2(c1(A))),('21',c1(c2(A))),('1+2',np.c_[c1(A),c2(A)])]:
   name=bn+pref+suff
   bases=[('D',I(D)),('zD',I(np.c_[z,D])),('AD',I(np.c_[A,D])),('zAD',I(np.c_[z,A,D]))];
   for name2,F in bases:
    cross=(z[:,:,None]*D[:,None,:]).reshape(len(z),-1); cr2=(A[:,:,None]*D[:,None,:]).reshape(len(z),-1)
    for bl,G in [('',F),('diag',np.c_[F,cross[:,::D.shape[1]+1][:,:4].sum(1)]),('crossZ',np.c_[F,cross]),('crossA',np.c_[F,cr2]),('quad',np.c_[F,D**2, *(([pair(D)] if D.shape[1]==4 else []))])]:
     for reg in [.001,.03,.3]:
      H=G[:,1:]; H=(H-H[:n].mean(0))/np.maximum(H[:n].std(0),1e-9); h=H[:n]; y=Y[:n].mean(0)+H[n:]@np.linalg.solve(h.T@h+reg*np.eye(h.shape[1]),h.T@(Y[:n]-Y[:n].mean(0))); er=np.abs(y-Y[n:]).mean(0); combos.append((er[0],er[1:].mean(),name,name2,bl,reg))
for k in [0,1]:
 print('readout',k)
 for a in sorted(combos,key=lambda q:q[k])[:18]:print(a)
np.savez('/tmp/coords.npz',z=z)
