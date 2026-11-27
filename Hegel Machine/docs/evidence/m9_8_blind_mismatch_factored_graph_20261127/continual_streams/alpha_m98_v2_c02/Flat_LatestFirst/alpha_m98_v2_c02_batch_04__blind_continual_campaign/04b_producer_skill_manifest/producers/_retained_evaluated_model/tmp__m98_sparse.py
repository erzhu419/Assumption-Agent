import pickle,numpy as np
from itertools import combinations_with_replacement
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; L=C['true_channel_loads']; R=C['bridge_raw']; N=len(L)
np.set_printoptions(precision=6,suppress=True)
def features(x): return np.c_[np.ones(len(x)),x,x**3,x*np.roll(x,-1,axis=1)]
X=features(L)
A=np.linalg.lstsq(X,R,rcond=None)[0]; E=R-X@A; e=np.linalg.norm(E,axis=1); keep=np.ones(N,dtype=bool); keep[np.argsort(e)[-3:]]=False
print('largest residual before trim',[(int(i),float(e[i]),L[i].tolist()) for i in np.argsort(e)[-12:]])
for mode in ('shared','sparse'):
 pred=np.zeros_like(R); mats=[]
 for j in range(4):
  ix=np.arange(13) if mode=='shared' else np.r_[np.arange(5),5+j,9+j]; U=X[:,ix]; b=np.linalg.lstsq(U[keep],R[keep,j],rcond=None)[0]; pred[:,j]=U@b; h=(np.linalg.qr(U[keep])[0]**2).sum(1); mats.append(h)
 E=R-pred; print(mode,'largest clean row errors',np.sort(np.linalg.norm(E[keep],axis=1))[-8:],'clean residual singular values',np.linalg.svd(E[keep],compute_uv=False)); print('sweep outlier residuals',E[~keep]); print('clean LOO max per channel',np.max(abs(E[keep]/(1-np.array(mats).T)),axis=0)); print('coefficients',np.linalg.lstsq(X[keep],R[keep],rcond=None)[0]);
print('row IDs of single-channel sweeps',np.flatnonzero((L!=0).sum(1)==1)); print('test/train raw max norm',[(k,max(np.linalg.norm(D[k]['past_channel_raw'],axis=-1).max(),np.linalg.norm(D[k]['hold_channel_raw'],axis=-1).max())) for k in ('train','test')]);
