import pickle, numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,t=tr['x'],tr['y'],tr['traj']; ts=s['eval_times']; np.set_printoptions(precision=5,suppress=True)
def aff(p,k,trim=False):
 d=s[p]; u,v=d['u'],d[k]; a=np.c_[u,np.ones(len(u))]; i=np.arange(len(u));
 if trim:
  for _ in range(5):
   w=np.linalg.lstsq(a[i],v[i],rcond=None)[0]; i=np.argsort(np.linalg.norm(a@w-v,axis=1))[:len(u)-3]
 w=np.linalg.lstsq(a[i],v[i],rcond=None)[0]; print(p,k,w,'rms',np.std(a[i]@w-v[i])); return w
A=aff('panel_a','v',True); B=aff('panel_b','v'); C=aff('panel_c','v1'); print('times telemetry summary',ts,s['telemetry_vec'],tr['summary'])
def basis(r):return -np.expm1(-ts[:,None]*r)
def fun(r):
 p=basis(r);return (p@np.linalg.lstsq(p,t.T,rcond=None)[0]-t.T).ravel()
r=least_squares(fun,[1.5,.25],bounds=([.05,.01],[8,5])).x; print('rates',r,'trajectory rms',np.std(fun(r))); T=np.c_[y,t@np.linalg.qr(basis(r))[0]]
def mono(z,ds):return np.column_stack([np.prod(z[:,i],axis=1) for d in ds for i in combinations_with_replacement(range(4),d)])
def val(P,alpha=0):
 P=np.c_[np.ones(len(x)),P]; P/=np.maximum(P[:52].std(0),.01);a=P[:52]; c=np.linalg.solve(a.T@a+alpha*np.eye(a.shape[1]),a.T@T[:52]) if alpha else np.linalg.lstsq(a,T[:52],rcond=None)[0];print(np.mean(abs(P[52:]@c-T[52:]),0),end=' ')
for name,z in [('x',x),('Ax',x@A[:4]+A[4]),('Ai',(x-A[4])@np.linalg.inv(A[:4])),('Bx',x@B[:4]+B[4]),('Bi',(x-B[4])@np.linalg.inv(B[:4])),('ABx',(x@A[:4]+A[4])@B[:4]+B[4]),('AiBx',((x-A[4])@np.linalg.inv(A[:4]))@B[:4]+B[4]),('Cx',x@C[:4]+C[4])]:
 print(name, end=' '); q=mono(z,[1,2]); val(q);val(mono(z,[1,2,3]),.01); val(np.c_[q,np.tanh(z),np.sin(2*z),np.cos(2*z)],.01);val(np.c_[q,np.tanh(z),z*np.tanh(z),np.tanh(2*z),z*np.tanh(2*z)],.03);print()
