import numpy as np, pickle
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
c=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']; X=c['true_channel_loads']; Y=c['bridge_raw']
valid=np.max(abs(Y),axis=1)<6
folds=np.array_split(np.random.default_rng(23).permutation(np.flatnonzero(valid)),5)
def design(x,j,k):
 z=x[:,j]; Q=[x[:,a]*x[:,b] for a,b in combinations_with_replacement(range(4),2)]; C=[x[:,a]*x[:,b]*x[:,d] for a,b,d in combinations_with_replacement(range(4),3)]; base=[np.ones(len(x)),*x.T]; extras={'lin':[], 'diag':[z**3], 'pair':Q+[z**3], 'joint':Q+C, 'odd':[v**3 for v in x.T]+[x[:,a]*x[:,b] for a in range(4) for b in range(a+1,4)], 'mixpoly':Q+[v**3 for v in x.T], 'odd3':C, 'by_channel':Q+[z*z*v for v in x.T]+[z*v*v for i,v in enumerate(x.T) if i!=j]};return np.column_stack(base+extras[k])
for key in ['lin','diag','pair','joint','odd','mixpoly','odd3','by_channel']:
 e=np.zeros_like(Y); E=np.zeros_like(Y)
 for j in range(4):
  a=design(X,j,key); b=np.linalg.lstsq(a[valid],Y[valid,j],rcond=None)[0]; E[:,j]=a@b-Y[:,j]
  for ids in folds:
   keep=np.setdiff1d(np.flatnonzero(valid),ids); b=np.linalg.lstsq(a[keep],Y[keep,j],rcond=None)[0]; b=least_squares(lambda v:a[keep]@v-Y[keep,j],b,loss='soft_l1',f_scale=.005).x;e[ids,j]=a[ids]@b-Y[ids,j]
 print(key,'oof',np.quantile(abs(e[valid]),[.5,.9],axis=0).round(5),'in rms',np.sqrt(np.mean(E[valid]**2,axis=0)).round(5))
 if key=='joint':
  for j in range(4):
   a=design(X,j,key);b=np.linalg.lstsq(a[valid],Y[valid,j],rcond=None)[0];print('joint coef',j,b.round(3))
