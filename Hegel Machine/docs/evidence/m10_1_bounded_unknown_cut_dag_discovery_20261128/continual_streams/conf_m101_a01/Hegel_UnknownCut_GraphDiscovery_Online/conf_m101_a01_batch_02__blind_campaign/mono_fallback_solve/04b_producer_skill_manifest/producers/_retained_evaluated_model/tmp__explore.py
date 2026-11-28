import numpy as np,pickle,itertools
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['train']; x,y,z=d['x'],d['y'],d['traj']; t=s['eval_times']; target=np.c_[y,z]
pa,pb,pc=[s['panel_'+j] for j in 'abc']
def aff(p,n='v',rob=False):
 a=np.c_[np.ones(len(p['u'])),p['u']]; v=p[n]; w=np.ones(len(a));
 for i in range(15 if rob else 1):
  b=np.linalg.lstsq(a*w[:,None],v*w[:,None],rcond=None)[0]; e=np.linalg.norm(a@b-v,axis=1); w=np.minimum(1,2*np.median(e)/np.maximum(e,1e-12))
 return b
A=aff(pa,rob=True); A[1:]=np.diag(np.diag(A[1:])); B=aff(pb); C=aff(pc,'v1')
def tf(x,b):return x@b[1:]+b[0]
coords={'x':x,'A':tf(x,A),'Ai':(x-A[0])@np.linalg.inv(A[1:]),'B':tf(x,B),'AB':tf(tf(x,A),B),'AiB':tf((x-A[0])@np.linalg.inv(A[1:]),B),'AC':tf(tf(x,A),C),'AiC':tf((x-A[0])@np.linalg.inv(A[1:]),C)}
if __name__=='__main__':
 np.set_printoptions(precision=4,suppress=True,linewidth=150)
 _,sv,V=np.linalg.svd(z,full_matrices=False); F=z@V[:2].T; Y=np.c_[y,F]; print('t, V',t,V[:2]); print('F correlation',np.corrcoef(np.c_[x,Y].T)[-3:]); print('std',Y.std(0));
 for name,q in coords.items():
  f=PolynomialFeatures(2).fit_transform(q); b=np.linalg.lstsq(f,Y,rcond=None)[0]; print(name,'quad coefficients\n',b.T); 
  if name=='x':
   for j in range(3):
    m=np.zeros((4,4)); m[np.triu_indices(4)]=b[5:,j]; m=(m+m.T)/2; print('eig',np.linalg.eigh(m));
 print('train summary actual',y.mean(),y.std(),z[:,-1].mean(),z[:,-1].std()); print('target correlations with direct poly fit residual',np.corrcoef(np.c_[Y,Y-f@b].T));
