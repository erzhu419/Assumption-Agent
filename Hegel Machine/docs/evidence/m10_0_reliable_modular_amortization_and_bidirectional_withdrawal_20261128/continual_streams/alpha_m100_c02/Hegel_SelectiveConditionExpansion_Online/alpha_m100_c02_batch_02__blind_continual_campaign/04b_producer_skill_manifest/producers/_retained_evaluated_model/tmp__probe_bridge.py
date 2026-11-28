import pickle,numpy as np
from itertools import combinations_with_replacement
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=p['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw'];
def basis(x,name):
 z=[np.ones(len(x)),*x.T]; pairs=[x[:,i]*x[:,j] for i,j in combinations_with_replacement(range(4),2)]
 if name=='aff': return np.array(z).T
 if name=='diag2': return np.c_[np.array(z).T,x*x]
 if name.startswith('poly'):
  for d in range(2,int(name[-1])+1): z.extend(np.prod(x[:,ix],axis=1) for ix in combinations_with_replacement(range(4),d))
 else:
  z+=pairs
  k,f=name.split(':'); a=float(k); z.extend({'sin':lambda:np.sin(a*x),'tanh':lambda:np.tanh(a*x),'rat':lambda:x/(1+(a*x)**2),'cub':lambda:x**3}[f]().T)
 return np.array(z).T

def fit(z,y):
 w=np.ones(len(z));
 for _ in range(16):
  b=np.linalg.lstsq(z*w[:,None],y*w[:,None],rcond=1e-10)[0]; e=np.linalg.norm(z@b-y,axis=1); w=1/np.sqrt(1+(e/max(.3*np.median(e),1e-7))**2)
 keep=e<max(6*np.median(e),.15); return np.linalg.lstsq(z[keep],y[keep],rcond=1e-10)[0]
if __name__=='__main__':
 for name in ['aff','diag2','poly2','poly3','1:cub']+[f'{a}:{f}' for f in ('sin','tanh','rat') for a in (.5,1,1.5,2)]:
  z=basis(x,name); err=[]
  for i in range(len(x)):
   m=np.arange(len(x))!=i; err.append(z[i]@fit(z[m],y[m])-y[i]);
  e=np.array(err); n=np.linalg.norm(e,axis=1); clean=n<5; axis=np.count_nonzero(x,axis=1)==1; print(name,'n',clean.sum(),'axis/mixed max',np.round([max(n[clean&axis]),max(n[clean&~axis])],5),'median',round(np.median(n),5),'channel RMSE',np.round(np.sqrt(np.mean(e[clean]**2,axis=0)),5),'gross',np.where(~clean)[0]);
 print('quadratic coefficients');print(np.round(fit(basis(x,'poly2'),y),5)); print('cubic coefficients'); print(np.round(fit(basis(x,'poly3'),y)[15:],5))
