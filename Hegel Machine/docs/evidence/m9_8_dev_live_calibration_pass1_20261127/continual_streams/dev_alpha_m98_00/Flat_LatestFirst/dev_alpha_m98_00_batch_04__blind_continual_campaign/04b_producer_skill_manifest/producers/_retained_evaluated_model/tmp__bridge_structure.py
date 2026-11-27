import numpy as np,pickle
from scipy.optimize import least_squares
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=D['calibration_load_cells']; L=c['true_channel_loads']; R=c['bridge_raw']
single=(L!=0).sum(1)==1
A=lambda x:np.c_[np.ones(len(x)),x,x*x,x**3]
B=np.linalg.lstsq(A(L[single]),R[single],rcond=None)[0]; E=R-A(L)@B
print('axis count',single.sum(),'axis residual max',abs(E[single]).max()); print('axis coefficients intercept, linear, quadratic, cubic\n',B.round(6))
print('all mixed rows: row, loads, residual after axis model')
for i in np.flatnonzero(~single): print(i,L[i].round(4),E[i].round(5))
pairs=[(i,j) for i in range(4) for j in range(i+1,4)]; P=np.column_stack([L[:,i]*L[:,j] for i,j in pairs]);
for c in range(4):
 fit=least_squares(lambda b:P@b-E[:,c],np.zeros(6),loss='cauchy',f_scale=.005); e=E[:,c]-P@fit.x
 print('pair model output',c,'coef',fit.x.round(6),'abs err quantiles',np.percentile(abs(e),[25,50,75,90,95,100]).round(7),'large rows',np.flatnonzero(abs(e)>.025))
