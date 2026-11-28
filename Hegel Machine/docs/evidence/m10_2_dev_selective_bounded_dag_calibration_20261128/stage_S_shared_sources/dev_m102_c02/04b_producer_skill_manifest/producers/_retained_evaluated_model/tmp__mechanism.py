import sys,pickle,numpy as np
from scipy.linalg import polar
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from batch_solver import _affine,_quadratic
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['train']; x,y,z=(d[k] for k in ('x','y','traj')); t=s['eval_times']; n=len(x); X=_affine(x); P=_quadratic(x)
np.set_printoptions(precision=5,suppress=True,linewidth=140)
q=np.linalg.lstsq(P,y,rcond=None)[0]; H=np.zeros((4,4)); H[np.triu_indices(4)]=q[5:]; H=(H+H.T)/2; print('keys',s.keys(),d.keys(),'telemetry',s['telemetry_vec']); print('H raw',H)
a=s['panel_a']; keep=np.linalg.norm(a['v']-a['u'],axis=1)<2; ca=np.linalg.lstsq(_affine(a['u'][keep]),a['v'][keep],rcond=None)[0]; print('A',ca)
for k in ('panel_b','panel_c'):
 a=s[k]; u=a['u'];
 for name in set(a)-{'u'}:
  fit=np.linalg.lstsq(_affine(u),a[name],rcond=None)[0]; fitq=np.linalg.lstsq(_quadratic(u),a[name],rcond=None)[0]; print(k,name,'affine/quadratic RMS',np.mean((a[name]-_affine(u)@fit)**2)**.5,np.mean((a[name]-_quadratic(u)@fitq)**2)**.5); print('singular values',np.linalg.svd(fit[1:],compute_uv=False));
  for tag,T in [('fwd',fit[1:]),('rev',fit[1:].T),('orth',polar(fit[1:])[0]),('orthrev',polar(fit[1:])[0].T)]:
   for at,A in [('raw',np.eye(4)),('cal',ca[1:]),('uncal',np.linalg.inv(ca[1:]))]:
    B=np.linalg.inv(A@T); print(tag,at,'H normalized',np.round((B@H@B.T)*100,2));
print('singular values after decay fits'); W=np.linalg.lstsq(X,z,rcond=None)[0];
for variant in ('exp2','exp1linear','exp2linear','exp2const'):
 def basis(p):
  b=-np.expm1(-np.exp(p[:,None])*t)
  return np.r_[b, [t]] if variant.endswith('linear') else np.r_[b,[np.ones_like(t)]] if variant.endswith('const') else b
 k=1 if variant=='exp1linear' else 2
 def loss(p):
  B=basis(p); return (np.linalg.lstsq(B.T,W.T,rcond=None)[0].T@B-W).ravel()
 f=least_squares(loss,np.log([.5,2][:k]),max_nfev=100); print(variant,np.exp(f.x),np.linalg.norm(f.fun),np.linalg.svd(basis(f.x),compute_uv=False))
