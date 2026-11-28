import sys,pickle,numpy as np,itertools
sys.path.insert(0,'/holdout_task'); from batch_solver import *
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; z=align(s['panel_a'],tr['x']); w,c,d=features(s,z); t=s['eval_times']; T=tr['traj']; y=tr['y']; a=slice(0,52); h=slice(52,None); np.set_printoptions(precision=3,suppress=True)
F=-np.expm1(-t[:,None]*[.37633,1.78968,5.5352]); H=ls(F,T.T).T; Y=np.c_[y,H]; print('H3 corrs',np.corrcoef(Y.T));
inds=[()]+[j for m in range(1,4) for j in itertools.combinations_with_replacement(range(4),m)]; P=np.column_stack([z[:,j].prod(1) for j in inds]); co=ls(P[a],Y[a]); print('cubic H3',[(j,round(k,3)) for j,k in zip(inds,co[:,3]) if abs(k)>.02]);
# Sweep groupwise additive rather than high-dimensional full interactions.
blocks={'z':z,'p':pair(z),'q':np.cos(2.4*z),'w':w,'c':c,'d':d,'cw':pair(w),'dw':np.cos(2.4*w)}
for k in range(4):
 print('out',k)
 for label,X in blocks.items():
  for typ,fun in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('r',lambda q:q/(1+q*q))]:
   full=[]
   for f in [.5,1,1.25,1.5,2,2.4,3,4,5,6]:
    cols=one(np.c_[z,pair(z),np.cos(2.4*z),fun(f*X)]); p=cols@ls(cols[a],Y[a,k]); full.append((abs(p[h]-Y[h,k]).mean(),f,abs(p[a]-Y[a,k]).mean()))
   b=min(full)
   if b[0]<[.43,.36,.20,.085][k]:print(label,typ,b)
# See whether residuals of the three laws collapse into shared directions after linear physical terms.
for X in [one(z),one(np.c_[z,pair(z),np.cos(2.4*z)]),P[:,:15],P]:
 E=Y-X@ls(X,Y); print('residual sv',np.linalg.svd(E,full_matrices=False)[1:]);
