import pickle,numpy as np
from scipy.optimize import least_squares
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; X=C['true_channel_loads']; Y=C['bridge_raw']
F={'linear':lambda x: [], 'square':lambda x:[x*x], 'signed_square':lambda x:[x*abs(x)], 'cubic':lambda x:[x**3], 'abs':lambda x:[abs(x)], 'mixed_square':lambda x:[x*x,x*abs(x)], 'odd_float':lambda x:[np.sign(x)*abs(x)**1.5], 'hinge':lambda x:[np.sign(x)*np.maximum(abs(x)-.5,0)]}
def design(x,j,k):return np.column_stack([np.ones(len(x)),x,*F[k](x[:,j])])
def fit(a,y):
 b=np.linalg.lstsq(a,y,rcond=None)[0]
 for scale in (.5,.12,.025,.004,.001):
  b=least_squares(lambda b:a@b-y,b,jac=lambda b:a.copy(),loss='soft_l1',f_scale=scale).x
 return b
if __name__=='__main__':
 for key in F:
  oof=np.empty_like(Y); coeff=[]
  for j in range(4):
   a=design(X,j,key); coeff.append(fit(a,Y[:,j]));
   for fold in np.array_split(np.random.default_rng(23).permutation(len(X)),5):
    keep=np.setdiff1d(np.arange(len(X)),fold);oof[fold,j]=a[fold]@fit(a[keep],Y[keep,j])-Y[fold,j]
  print(key,'per channel q50/q90/q95',np.quantile(abs(oof),[.5,.9,.95],axis=0).round(5).tolist());
  if key in ('signed_square','cubic'):
   print('coef',np.asarray(coeff).round(5));e=np.column_stack([design(X,j,key)@coeff[j]-Y[:,j] for j in range(4)]); print('worst per channel',[list(zip(np.argsort(abs(v))[-6:],np.sort(abs(v))[-6:].round(4))) for v in e.T]); print('sweeps residual',e[:16].round(4).tolist())
