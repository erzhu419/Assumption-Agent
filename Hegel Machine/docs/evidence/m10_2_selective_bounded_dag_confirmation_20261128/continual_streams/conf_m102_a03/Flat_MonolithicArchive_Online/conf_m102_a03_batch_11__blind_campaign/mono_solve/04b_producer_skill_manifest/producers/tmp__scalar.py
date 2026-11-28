import pickle,numpy as np,runpy
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
np.set_printoptions(precision=5,suppress=True,linewidth=150)
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,T=(S['train'][k] for k in ('x','y','traj')); a=runpy.run_path('/tmp/hegel_continual_archive/conf_m102_a03_batch_09_solver.py'); z=a['_decode'](x,a['_calibration'](S['panel_a'])); aff=a['_affine']; p=S['panel_b']; w=aff(z)@a['_fit'](aff(p['u']),p['v']);
def fit(F,Y=y): return np.linalg.lstsq(F[:52],Y[:52],rcond=None)[0]
def score(P): return np.round([np.mean(abs(P[:52]-y[:52])),np.mean(abs(P[52:]-y[52:])),np.max(abs(P[52:]-y[52:]))],5)
for name,u in [('raw',x),('z',z),('w',w)]:
 A=aff(u); Q=np.array([u[:,i]*u[:,j] for i,j in combinations_with_replacement(range(4),2)]).T; C=np.array([u[:,i]*u[:,j]*u[:,k] for i,j,k in combinations_with_replacement(range(4),3)]).T; pair=(u[:,0]*u[:,2]+u[:,1]*u[:,3])[:,None];
 for label,F in [('lin',A),('pair',np.c_[A,pair]),('quad',np.c_[A,Q]),('cubic',np.c_[A,Q,C])]:
  c=fit(F); P=F@c; print(name,label,score(P),c if label!='cubic' else '');
  if label=='pair' and name=='w': print('y pred resid trajectory end',np.c_[y,P,y-P,T[:,-1]][np.argsort(P)][::2]);
  if name=='raw': continue
  for f in [lambda v:np.tanh(v),lambda v:np.arctan(v),lambda v:np.sin(v),lambda v:v/np.sqrt(1+v*v)]:
   def pred(p): return p[-2]+p[-1]*f(F@p[:-2]);
   m=least_squares(lambda p:(pred(p)-y)[:52],np.r_[c/2,0,2],max_nfev=3000); print('   ',f(1.),score(pred(m.x)),m.x if label in ['lin','pair'] else '')
