import pickle, numpy as np, importlib.util
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']
sp=importlib.util.spec_from_file_location('a','/tmp/hegel_continual_archive/dev_m102_c02_batch_06_solver.py'); a=importlib.util.module_from_spec(sp);sp.loader.exec_module(a)
z=a._calibration(s['panel_a'],tr['x'])
def aff(u):return np.c_[np.ones(len(u)),u]
def fit(h,v):return np.linalg.lstsq(h,v,rcond=None)[0]
forms={ 'lin':(lambda u,p:u,[]),'pow':(lambda u,p: np.sign(u)*abs(u)**p[0],[2]),'hinge':(lambda u,p: np.sign(u)*np.maximum(abs(u)-p[0],0)**p[1],[.25,2]),'rational':(lambda u,p: u/(1+p[0]*u**2),[.22]),'cubic':(lambda u,p:u+p[0]*u**3,[.22]),'sat':(lambda u,p:np.tanh(p[0]*u),[2]),'powered_rat':(lambda u,p:np.sign(u)*abs(u)**p[0]/(1+p[1]*abs(u)**p[2]),[2,.22,2]),'sine':(lambda u,p:np.sin(p[0]*u),[2]),'blend_sin':(lambda u,p:u+p[0]*np.sin(p[1]*u),[.22,2]),'blend_tanh':(lambda u,p:u+p[0]*np.tanh(p[1]*u),[.22,2])}
if __name__=='__main__':
 print('calib',z[:4],z.min(0),z.max(0)); print('svd',np.linalg.svd(tr['traj'])[1]); print('y/traj correlation',np.corrcoef(np.c_[tr['y'],tr['traj']].T)[0]);
 for pn,k in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
  u,v=s[pn]['u'],s[pn][k]; print(pn,k)
  for name,(fn,p0) in forms.items():
   def fun(p):
    h=aff(fn(u,p)); return (h@fit(h,v)-v).ravel()
   p=least_squares(fun,p0,max_nfev=150,bounds=([0]*len(p0),[20]*len(p0))).x if p0 else []
   print(name,np.round(p,5),np.sqrt(np.mean(fun(p)**2)))
  for h in [aff(np.c_[u,u**2,u**3]),aff(np.c_[u,a._quad(u)]),aff(np.c_[u,a._quad(u),u**3]),aff(np.c_[u,u*np.roll(u,-1,axis=1),u**3])]:
   print('poly',h.shape[1],np.sqrt(np.mean((h@fit(h,v)-v)**2)))
