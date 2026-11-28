from graphprobe import *
from itertools import combinations_with_replacement as cr
T=S['train']; y=T['y']; h=T['traj']; t=S['eval_times']; ai,af,B,C,D=ops(S); z=ai(T['x']); c,d=C(z),D(z); b=B(z)
def p(x,k):
 i=list(cr(range(x.shape[1]),k));return np.stack([np.prod(x[:,j],axis=1) for j in i],axis=1),i
P2,I2=p(z,2); P3,I3=p(z,3); dot=(c*d).sum(1); X=one(np.c_[P2,P3,c,d,dot]); Y=np.c_[y,h]; W=fit(X,Y); E=Y-X@W
names=['1']+['z'+''.join(map(str,q)) for q in I2+I3]+[f'{a}{j}' for a in 'cd' for j in range(4)]+['dot']; print('base rms',E.std(0)); print('dot coefficients',W[-1]); print('coeffs');
for k,w in zip(names,W): print(k,w)
V=-np.expm1(-t[:,None]*np.array([.396,1.7987,5.27])); print('term profiles normalized at last time');
for nm,q in [('cubic',P3),('dot',dot[:,None]),('quad',P2),('cc',c),('dd',d),('resid',E[:,1:])]:
 if nm=='resid': R=q
 else:
  L=np.c_[X,q]; inds=np.arange(len(X.T));
  key={'cubic':np.arange(11,31),'dot':[39],'quad':np.arange(1,11),'cc':np.arange(31,35),'dd':np.arange(35,39)}[nm]; G=X[:,np.setdiff1d(inds,key)]; R=h-G@fit(G,h); q=q-G@fit(G,q); R=q@fit(q,R)
 _,s,vt=np.linalg.svd(R,full_matrices=False); print(nm,s[:5],vt[:3]/vt[:3,-1:]);
 for label,f in [('exp',lambda k:-np.expm1(-k*t)),('rat',lambda k:t/(1+k*t)),('log',lambda k:np.log1p(k*t)),('power',lambda k:t**k),('exp-linear',lambda k:t*np.exp(-k*t)),('cos',lambda k:1-np.cos(k*t)),('tanh',lambda k:np.tanh(k*t))]:
  v=vt[0]/vt[0,-1]; r=minimize_scalar(lambda k:np.linalg.norm(v-f(k)/f(k)[-1]),bounds=(.005,5),method='bounded'); print(label,round(r.x,5),round(r.fun,6),end=' | ')
 print()
