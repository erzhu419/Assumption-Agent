import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations_with_replacement as comb
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; z=tr['traj']; t=s['eval_times']
def aff(u):return np.c_[np.ones(len(u)),u]
def poly(u,d):return np.column_stack([np.prod(u[:,idx],axis=1) for j in range(d+1) for idx in comb(range(u.shape[1]),j)])
def fit(f,v):return np.linalg.lstsq(f,v,rcond=None)[0]
a=s['panel_a']; f=aff(a['u']); m=np.ones(len(f),bool)
for _ in range(5):
 A=fit(f[m],a['v'][m]); e=np.linalg.norm(a['v']-f@A,axis=1);m=e<3*np.median(e)
b=s['panel_b']; B=fit(aff(b['u']),b['v']); q=(x-A[0])@np.linalg.inv(A[1:]); r=aff(q)@B
u,sv,v=np.linalg.svd(z,full_matrices=False); vv=z@v[:2].T
if __name__=='__main__':
 def wave(kind,p):
  if kind=='sin_pair':return np.c_[np.sin(p[0]*t),np.sin(p[1]*t)]
  if kind=='sin_slow':return np.c_[np.sin(p[0]*t),-np.expm1(-p[1]*t)]
  if kind=='sin_t':return np.c_[np.sin(p[0]*t),t]
  if kind=='sin_cos':return np.c_[np.sin(p[0]*t),1-np.cos(p[1]*t)]
  if kind=='damped_pair':return np.c_[np.exp(-p[2]*t)*np.sin(p[0]*t),np.exp(-p[3]*t)*np.sin(p[1]*t)]
  if kind=='sin_quad':return np.c_[np.sin(p[0]*t),t*t]
  if kind=='sin_tanh':return np.c_[np.sin(p[0]*t),np.tanh(p[1]*t)]
  if kind=='damped_slow':return np.c_[np.exp(-p[2]*t)*np.sin(p[0]*t),-np.expm1(-p[1]*t)]
 for k in ('sin_pair','sin_slow','sin_t','sin_cos','damped_pair','sin_quad','sin_tanh','damped_slow'):
  p0=[1.5,.6,.15,.15][:4 if k=='damped_pair' else 3 if k=='damped_slow' else 2 if k not in ('sin_t','sin_quad') else 1];
  def er(p):
   f=wave(k,p);return (f@fit(f,z[:56].T)-z[:56].T).ravel()
  p=least_squares(er,p0,bounds=(0,8),max_nfev=200).x; f=wave(k,p); print(k,np.round(p,5),np.std(er(p)),np.std(f@fit(f,z[56:].T)-z[56:].T))
 c=s['panel_c']; w=c['u'];
 for tag in ('v1','v2'):
  print(tag,'affine,quadratic,additive dictionaries');tar=c[tag];
  for k,f in [('aff',aff(w)),('quad',poly(w,2)),('adden',np.c_[aff(w),w*w,w**3,np.sin(2*w),np.cos(2*w)]),('add23',np.c_[aff(w),w*w,w**3]),('sin',np.c_[aff(w),np.sin(2*w),np.cos(2*w)]),('tanh',np.c_[aff(w),np.tanh(1.4*w),np.tanh(w)**2])]:
   h=f@np.linalg.pinv(f); err=(tar-h@tar)/np.maximum(1-h.diagonal()[:,None],1e-9);print(k,np.round(np.sqrt(np.mean(err**2,0)),5))
 for tag,w in [('raw',x),('cal',q),('rot',r),('rot_raw',aff(x)@B),('rev', (q-B[0])@np.linalg.inv(B[1:]))]:
  f=poly(w,2); res=np.c_[y,vv]-f@fit(f,np.c_[y,vv]); print(tag,'corr nonlinear residuals');
  dic=np.c_[w**3,w**4,np.sin(2*w),np.cos(2*w),np.tanh(w),np.tanh(w)**2,*[(w[:,i]*w[:,j])[:,None] for i,j in comb(range(4),2) if i!=j]];dic-=f@fit(f,dic); corr=dic.T@res/(np.maximum(np.linalg.norm(dic,axis=0),1e-14)[:,None]*np.linalg.norm(res,axis=0)); print(np.round(corr[:24],2))
