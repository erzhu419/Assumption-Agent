import pickle,numpy as np,importlib.util
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; n=52; times=s['eval_times']
def aff(x):return np.c_[np.ones(len(x)),x]
def fit(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def mod(i):
 sp=importlib.util.spec_from_file_location('arc',f'/tmp/hegel_continual_archive/dev_m102_c02_batch_{i:02}_solver.py'); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
A=s['panel_a']; u,v=A['u'],A['v']; keep=np.linalg.norm(v,axis=1)<4
for _ in range(3):
 M=fit(aff(u[keep]),v[keep]); r=np.linalg.norm(aff(u)@M-v,axis=1);keep=r<.4
print('A aﬀ err',np.sort(r),'M',M)
X=[('raw',x),('aff',(x-M[0])@np.linalg.inv(M[1:])),('A09',mod(9)._calibrate(A,x)),('A10',mod(10)._calibrate(A,x))]
U=mod(12); print('12 attrs',[q for q in dir(U) if not q.startswith('__')]);
if hasattr(U,'_calibrate'): X.append(('A12',(x-U._calibrate(A)[0])/U._calibrate(A)[1]))
PB=s['panel_b']; B=lambda z:aff(np.cos(2.6*z))@fit(aff(np.cos(2.6*PB['u'])),PB['v'])
P=s['panel_c']; R=lambda z:z/(1+1.07827*abs(z)**1.23306); S=lambda z:np.logaddexp(0,2.2*z); C=lambda z:aff(R(z))@fit(aff(R(P['u'])),P['v1']); D=lambda z:aff(S(z))@fit(aff(S(P['u'])),P['v2'])
F=-np.expm1(-times[:,None]*[.41318229,1.81372115]); T=fit(F,tr['traj'].T).T; targets=np.c_[tr['y'],T]
for tag,z in X:
 print('-----',tag,'zrange',z.min(),z.max())
 cand={'z':z,'b':B(z),'c':C(z),'d':D(z),'cosz':np.cos(2.6*z),'bc':C(B(z)),'bd':D(B(z)),'cb':B(C(z)),'db':B(D(z)),'cd':D(C(z)),'dc':C(D(z)),'b+d':B(z)+D(z),'b*c':B(z)*C(z),'b*d':B(z)*D(z),'b/d':B(z)/(1+.22*D(z)**2),'z*b':z*B(z)}
 for j,y in enumerate(targets.T):
  out=[]
  for k,q in cand.items():
   for w,H in [('lin',aff(q)),('poly2',aff(np.c_[q,*[q[:,a]*q[:,b] for a in range(4) for b in range(a,4)]]))]:
    pred=H@fit(H[:n],y[:n]);out.append((np.mean(abs(pred[n:]-y[n:])),np.mean(abs(pred[:n]-y[:n])),k,w))
  print(j,sorted(out)[:9])
np.savez('/tmp/stage15.npz',**dict(X),targets=targets)
