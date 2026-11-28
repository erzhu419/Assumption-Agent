import numpy as np,pickle
from scipy.sparse import diags, eye, kron
from scipy.optimize import minimize
from transport_model import _axis,_responses,_sweep_fit,_raw,_calibrate,_box_simplex
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; n,m=s['nx'],s['ny']; Z=s['Z_nuisance']; y=_calibrate(s); t=_sweep_fit(s); c=s['sensor_calibration_sweeps']; print('sweep coefficients',t.round(6)); print('raw inverse max error',np.max(abs(_raw(y.reshape(-1,4),t).ravel()-s['y_obs_raw'])),'calibrated range',y.min(),y.max());
for k in range(4):
 b=c['single_bank_sweeps'];use=b['bank_id']==k; xx=np.zeros((use.sum(),4));xx[:,k]=b['ref_signal'][use]; xx=np.vstack((c['ref_rosette'],xx)); yy=np.r_[c['raw_rosette'][:,k],b['raw_readout'][use]]; e=_raw(xx,t)[:,k]-yy; cv=e.copy()
 for j in range(5):
  mask=np.arange(len(e))%5==j;cv[mask]=_raw(xx[mask],_sweep_fit(s,j))[:,k]-yy[mask]
 print('bank',k,'outliers',np.flatnonzero(abs(e)>.001),'inlier CV max',np.max(abs(cv[abs(e)<.001])))
hx=s['Lx']/(n-1);hy=s['Ly']/(m-1); r=np.repeat([s['robin_x'],s['robin_y']],2); reaction=diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
def responses(p,wall=1.):
 ax=_axis(s['kappa_x'],s['vel_x'],hx,0.);ay=_axis(s['kappa_y'],s['vel_y'],hy,0.)
 if wall!=1.:
  for a in (ax,ay):
   for i,j in ((0,1),(-1,-2)):
    a[i,i]*=wall;a[i,j]*=wall
 dx=np.zeros(n);dy=np.zeros(m);dx[[0,-1]]=np.array(p[:2])/hx;dy[[0,-1]]=np.array(p[2:])/hy
 return _responses(s,kron(eye(m),ax+diags(dx),format='csc')+kron(ay+diags(dy),eye(n),format='csc')+reaction)
def fit(S, ids):
 a=np.column_stack((S[ids],y[ids]));z=Z[ids]; proj=np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(Z.shape[1]),z.T@a); a-=z@proj; x,b=a[:,:-1],a[:,-1]; ridge=s['source_ridge'];w,_=_box_simplex(x.T@x+ridge*np.eye(x.shape[1]),-x.T@b,s['u_box'],s['total_budget']);g=proj[:,-1]-proj[:,:-1]@w;return w,g,b-x@w
all_=np.ones(len(y),bool); groups=np.arange(len(y))//4; folds=[groups%8==i for i in range(8)]+[groups//8==i for i in range(3)]
def run(p,wall=1.,detail=False):
 S,U=responses(p,wall);w,g,e=fit(S,all_);v=e@e
 if detail:
  errors=[]
  for mask in folds:
   wc,gc,_=fit(S,~mask);errors.append(np.sqrt(np.mean((S[mask]@wc+Z[mask]@gc-y[mask])**2)))
  print('wall',wall,'p',np.round(p,4),'rmse',np.sqrt(v/len(y)),'cv avg,max,shift',np.mean(errors[:8]),max(errors[:8]),np.round(errors[8:],5),'w',w.round(4),'gamma',g.round(4),'field norm',np.linalg.norm(np.tensordot(w,U,axes=1)))
 return v
for p in [r,r*0,r*[1,1,0,0],r*[0,0,1,1],r*.5,r*2,r*[1,0,1,0],r*[0,1,0,1]]:run(p,detail=True)
for wall in (.5,1.,2.):
 f=minimize(lambda p:run(p,wall),r*.5,bounds=[(0,3)]*4,method='L-BFGS-B'); run(f.x,wall,True)
