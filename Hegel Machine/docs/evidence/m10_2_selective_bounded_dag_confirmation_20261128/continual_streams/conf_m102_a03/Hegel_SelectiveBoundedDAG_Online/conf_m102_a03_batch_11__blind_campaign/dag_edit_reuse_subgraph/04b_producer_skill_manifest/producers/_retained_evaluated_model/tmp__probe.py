import numpy as np,pickle,glob,runpy
from scipy.optimize import least_squares
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=S['train']['x']; y=S['train']['y']; t=S['train']['traj']; n=52
D=runpy.run_path(glob.glob('/tmp/hegel_continual_archive/*08_solver.py')[0]); fit=D['fit']; af=D['affine']; z=D['calibrate'](S['panel_a'],x)
from itertools import combinations_with_replacement as cwr

def poly(x,d):
 return np.column_stack([np.ones(len(x))]+[np.prod(x[:,p],axis=1) for j in range(1,d+1) for p in cwr(range(x.shape[1]),j)])
def score(name,A):
 if A.shape[1]>48:return
 a=A[:n]; b=A[n:]; Q,R=np.linalg.qr(a); u,s,v=np.linalg.svd(Q.T@t[:n],full_matrices=False); tc=fit(R,(u[:,:2]*s[:2])@v[:2]); yh=b@fit(a,y[:n]); th=b@tc
 print(name,A.shape[1],*(round(i,6) for i in (np.mean(abs(yh-y[n:])),np.mean(abs(th-t[n:])),np.mean(abs(a@fit(a,y[:n])-y[:n])),np.mean(abs(a@tc-t[:n])))))
print('traj sv',np.linalg.svd(t,compute_uv=False));print('tele',S['telemetry_vec'])
for key in ('panel_b','panel_c'):
 p=S[key];u=p['u'];print(key)
 for name,v in p.items():
  if name=='u':continue
  funcs={'lin':lambda x:x,'atan':lambda x:np.arctan(2.15*x),'sin':lambda x:np.sin(2.15*x),'tanh':lambda x:np.tanh(2.15*x),'rat':lambda x:x/(1+.22*x*x),'arch':D['_panel_b_warp'](p) if key=='panel_b' else lambda x:x}
  for f,F in funcs.items():
   A=af(F(u)); c=fit(A,v); b=af(F(z))@c; print(name,f,'panel rmse',np.mean((A@c-v)**2)**.5)
   score(f+' b',af(b)); score(f+' z,b',af(np.c_[z,b]));score(f+' b2',poly(b,2)); score(f+' z,b2',poly(np.c_[z,b],2));score(f+' b3',poly(b,3))
  for d in (1,2,3):
   A=poly(u,d); print(name,'poly panel',d,np.mean((A@fit(A,v)-v)**2)**.5)
score('z',af(z));score('z2',poly(z,2));score('z3',poly(z,3));
