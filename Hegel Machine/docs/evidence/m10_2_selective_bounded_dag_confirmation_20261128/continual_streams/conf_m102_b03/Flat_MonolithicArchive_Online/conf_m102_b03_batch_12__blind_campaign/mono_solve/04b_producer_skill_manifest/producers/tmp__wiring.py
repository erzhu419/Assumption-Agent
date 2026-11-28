import glob,importlib.util,itertools,pickle,numpy as np
from scipy.optimize import minimize_scalar,least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];tr=s['train'];n=52
spec=importlib.util.spec_from_file_location('arch',glob.glob('/tmp/hegel_continual_archive/*11_solver.py')[0]); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
D=m.D;ls=m.ls
u=s['panel_a']['u'];v=s['panel_a']['v'];ra=np.linalg.norm(D(u)@np.load('/tmp/core.npz')['cal']-v,axis=1);good=ra<1
print('arch calibration inlier absolute inversion error',np.quantile(abs(m.calibrate(s['panel_a'],v[good])-u[good]),[.5,.9,1]));z=m.calibrate(s['panel_a'],tr['x']);print('z range',z.min(0),z.max(0)); H,amp,rates,er=m.trajectory(s['eval_times'],tr['traj'][:n]);amp=ls(H,tr['traj'].T).T;Y=np.c_[tr['y'],amp];print('stats',rates,er,np.corrcoef(Y.T))
c=s['panel_c'];u=c['u'];pb=s['panel_b']; coefficients=ls(D(pb['u']),pb['v']); B=lambda x:D(x)@coefficients
quad=lambda x:D(x*np.roll(x,-1,axis=1)); r=ls(quad(u),c['v1']); P=lambda x:quad(x)@r
cf=lambda x,k:D(np.cos(k*x)); obj=lambda k:np.mean((cf(u,k)@ls(cf(u,k),c['v2'])-c['v2'])**2); k=minimize_scalar(obj,bounds=(1.7,3.2),method='bounded').x; w=ls(cf(u,k),c['v2']);Q=lambda x:cf(x,k)@w
print('panel residuals',np.std(P(u)-c['v1']),np.std(Q(u)-c['v2']))
fns={'B':B,'P':P,'Q':Q};A={'Z':z}
for d in range(3):
 for a,x in list(A.items()):
  if len(a)==d+1:
   for b,f in fns.items(): A[b+a]=f(x)
records=[]
def score(desc,F):
 pred=F@ls(F[:n],Y[:n]); train=np.mean(abs(pred[:n]-Y[:n]),0); er=np.mean(abs(pred[n:]-Y[n:]),0);records.append((desc,train,er))
for a,x in A.items():
 score(a,D(x))
 for b,h in A.items():
  if a>=b:continue
  G=np.c_[x,h]; X=(x[:,:,None]*h[:,None,:]).reshape(len(z),16);
  score(a+' '+b,D(G));score(a+' '+b+' dot',D(np.c_[G,np.sum(x*h,1)]));score(a+' '+b+' prod',D(np.c_[G,X]));
for j in range(3):
 print('BEST',j)
 for a,b,c in sorted(records,key=lambda r:r[2][j])[:15]:print(a,np.round(b,5),np.round(c,5))
np.savez('/tmp/features.npz',**A,Y=Y,H=H,rates=rates)
