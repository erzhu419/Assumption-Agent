import pickle,numpy as np,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=[s['panel_a'][k] for k in ('u','v')]; keep=np.max(abs(v),axis=1)<3; u,v=u[keep],v[keep]; print('clean rows',len(u),'telemetry',s['telemetry_vec']); np.set_printoptions(precision=4,suppress=True)
base=np.c_[np.ones(len(u)),u]; B=np.linalg.lstsq(base,v,rcond=None)[0]; E=v-base@B; print('affine',B,'rms',np.std(E,axis=0),'residual svd',np.linalg.svd(E,compute_uv=False)); print('rows [u, residual]',np.c_[u,E][:12]);
L=[]; names=[]
def add(n,q):
 if np.std(q)>1e-10: L.append(q); names.append(n)
for j in range(4):
 for k in range(j,4):
  for l in range(k,4): add(f'u{j}*u{k}*u{l}',u[:,j]*u[:,k]*u[:,l]);
  add(f'u{j}*u{k}',u[:,j]*u[:,k]);
for z in itertools.product([0,1,-1],repeat=4):
 if not any(z) or next(t for t in z if t)!=1: continue
 a=u@z
 for f in [.22,.6,1.,1.25,1.5,2.,2.15,2.5,3.,3.5,4.]:
  add(f'sin({f}*{z})',np.sin(f*a)); add(f'cos({f}*{z})',np.cos(f*a)); add(f'tanh({f}*{z})',np.tanh(f*a)); add(f'rat({f}*{z})',1/(1+f*a*a))
D=np.array(L).T; D-=base@np.linalg.lstsq(base,D,rcond=None)[0]; D/=np.maximum(1e-13,np.linalg.norm(D,axis=0));
for j in range(4):
 for mode in ['diag','full']:
  A=base if mode=='full' else base[:,[0,1+j]]; D0=np.array(L).T; D0-=A@np.linalg.lstsq(A,D0,rcond=None)[0]; D0/=np.maximum(1e-13,np.linalg.norm(D0,axis=0)); R=v[:,j]-A@np.linalg.lstsq(A,v[:,j],rcond=None)[0]; corr=D0.T@R/np.linalg.norm(R); ix=np.argsort(-abs(corr))[:6]; print(j,mode,[(names[k],round(corr[k],5)) for k in ix]); sel=[]; e=R
  for _ in range(3):
   k=np.argmax(abs(D0.T@e)); sel.append(k); M=D0[:,sel]; e=R-M@np.linalg.lstsq(M,R,rcond=None)[0]; print('   ',names[k], 'rms',np.sqrt(np.mean(e**2)))
for name in ('panel_b','panel_c'):
 p=s[name]; A=np.c_[np.ones(len(p['u'])),p['u']]; print(name,[(k,np.linalg.svd(np.linalg.lstsq(A,v,rcond=None)[0][1:],compute_uv=False)) for k,v in p.items() if k!='u'])
