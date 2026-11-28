import pickle,numpy as np
from itertools import combinations, combinations_with_replacement
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; X=C['true_channel_loads']; Y=C['bridge_raw']; S=C['single_channel_sweeps']
P=list(combinations(range(4),2)); Q=list(combinations_with_replacement(range(4),3))
def design(x,k):
 cols=[np.ones(len(x)),*x.T]; cols+=list((x*x).T) if k in (1,2,3,4) else []; cols+=list((x**3).T) if k in (1,2,3) else []
 if k in (2,3,4): cols += [x[:,i]*x[:,j] for i,j in P]
 if k==3: cols += [x[:,i]*x[:,j]*x[:,h] for i,j,h in Q if i!=h]
 return np.column_stack(cols)
def fit(A,y):
 b=np.linalg.lstsq(A,y,rcond=None)[0]
 for _ in range(50):
  r=np.linalg.norm(A@b-y,axis=1); s=max(1e-4,np.median(r)); w=1/np.hypot(1,r/(2*s)); b=np.linalg.lstsq(A*w[:,None],y*w[:,None],rcond=None)[0]
 r=np.linalg.norm(A@b-y,axis=1); good=r<max(.002,6*np.median(r)); return np.linalg.lstsq(A[good],y[good],rcond=None)[0]
for j in range(4):
 z=S['lab_id']==j; t=S['true_stress'][z]; v=S['cmd_stress'][z]; print('sweep',j,[(k,np.max(abs(np.polyval(np.polyfit(t,v,k),t)-v))) for k in range(1,6)],np.polyfit(t,v,3))
for k in (0,4,1,2,3):
 A=design(X,k); b=fit(A,Y); r=np.linalg.norm(A@b-Y,axis=1); print('model',k,'residual',np.percentile(r,[50,80,90,95,99,100]),'large',np.flatnonzero(r>.1),'rank',np.linalg.matrix_rank(A))
 if k==2: print('coeff',np.round(b,6)); print('sorted rows',np.round(np.sort(r),5))
print('calibration design: axis rows',np.flatnonzero(np.count_nonzero(X,axis=1)==1),'largest mixed abs coordinate',abs(X[np.count_nonzero(X,axis=1)>1]).max())
