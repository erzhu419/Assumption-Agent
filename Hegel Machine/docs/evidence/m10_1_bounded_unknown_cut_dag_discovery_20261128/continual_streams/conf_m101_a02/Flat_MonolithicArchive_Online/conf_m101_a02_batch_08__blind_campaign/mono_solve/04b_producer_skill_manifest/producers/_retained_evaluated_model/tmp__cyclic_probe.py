from campaign_probe import *
from itertools import combinations
np.set_printoptions(precision=5,suppress=True,linewidth=140)
z,b,c,d=(a[:n] for a in old._coords(s)); y=tr['y']; T=tr['traj']; j=np.arange(4); h=(j-1)%4; P=z*b[:,h]; a=P.sum(1); Q=np.c_[aff(z),b,c,d]; print('cyclic contrasts',np.corrcoef(np.c_[P,y,T[:,-1]].T)[-2:]); print('telemetry',s['telemetry_vec']);
for l in (52,56):
 print('split',l)
 def score(F,Y):
  coef=fit(F[:l],Y[:l]); return np.mean(abs(F[l:]@coef-Y[l:]),axis=0),np.std(F[:l]@coef-Y[:l],axis=0),coef
 R=T[:l]-Q[:l]@fit(Q[:l],T[:l]); V=np.linalg.svd(R,full_matrices=False)[2][0]*-1; print('projection',V/V[-1]);
 for p in [1,1.2,1.35,1.5,1.75,2]:
  H=np.log1p(p*times); HH=np.c_[np.ones(len(times)),H]; print('log',p,np.linalg.norm(V-HH@fit(HH,V)))
 for label,F in [('cyclic',np.c_[aff(b),P]),('cyclic+z',np.c_[aff(b),z,P]),('cyclic+z+c',np.c_[aff(b),z,c,P]),('cyclic+z+d',np.c_[aff(b),z,d,P]),('cyclic+z+c+d',np.c_[Q,P]),('shared',np.c_[Q,a]),('shared+bquad',np.c_[Q,a,*[b[:,i]*b[:,j] for i in range(4) for j in range(i,4)]]),('shared+bdiag',np.c_[Q,a,b*b]),('shared+zquad',np.c_[Q,a,*[z[:,i]*z[:,j] for i in range(4) for j in range(i,4)]]),('shared+near',np.c_[Q,a,z*np.roll(z,-1,axis=1),b*np.roll(b,-1,axis=1)])]:
  err,sd,w=score(F,np.c_[y,T]); print(label,err[[0,1,8]],sd[[0,1,8]],'mean trajectory',err[1:].mean());
  if label=='shared':print('coef y',w[:,0]); print('T basis coefficients',w[[0,1,5,9,13,-1],1:]); print('y contrast',y[:12]-F[:12]@w[:,0])
 # use the projection to extract a unit-scale amplitude without leaking validation states
 coeff=fit(np.c_[Q[:l],a[:l]],T[:l])[-1]; residual=(T[:l]-T[:l].mean(0))@coeff/(coeff@coeff); print('amplitude slopes',fit(np.c_[Q[:l],residual],y[:l])[-1]); print('trajectory coef',coeff)
