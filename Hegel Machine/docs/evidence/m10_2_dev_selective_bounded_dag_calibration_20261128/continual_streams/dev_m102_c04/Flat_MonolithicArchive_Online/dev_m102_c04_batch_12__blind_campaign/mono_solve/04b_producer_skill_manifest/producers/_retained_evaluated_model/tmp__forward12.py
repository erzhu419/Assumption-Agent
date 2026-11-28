import sys,pickle,itertools,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
from dev_m102_c04_batch_11_solver import _calibrate
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; tt=s['eval_times']; n=52
D=lambda *a: np.column_stack((np.ones(len(a[0])),*a))
def ls(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
z=_calibrate(s['panel_a'],t['x']); b,c=s['panel_b'],s['panel_c']; B=D(z)@ls(D(b['u']),b['v']);
pairs=list(itertools.combinations(range(4),2)); pp=lambda x: np.column_stack([x[:,i]*x[:,j] for i,j in pairs]); co=ls(D(pp(c['u'])),c['v1']); print('pair weights',co)
P=D(pp(z))@co; f=lambda a:D(np.cos(2.4*a)); R=f(z)@ls(f(c['u']),c['v2']);
A=t['traj']; H=lambda k:-np.expm1(-tt[:,None]*k); rr=lambda k:(H(k)@ls(H(k),A[:n].T)-A[:n].T).ravel(); opt=least_squares(rr,[1.7,.45],bounds=([.01,.01],[5,5])); h=H(opt.x); Y=np.column_stack((t['y'],ls(h,A.T).T)); print('temporal',opt.x,np.std(opt.fun), 'amp std',Y.std(0));
for name,x in [('B',B),('P',P),('R',R),('z',z),('Bz',D(B,z)),('PR',D(P,R)),('BP',D(B,P)),('BR',D(B,R)),('BPR',D(B,P,R))]:
 def stats(f):
  pred=f@ls(f[:n],Y[:n]); return np.round(np.mean(abs(pred[:n]-Y[:n]),0),5),np.round(np.mean(abs(pred[n:]-Y[n:]),0),5),round(np.mean(abs(pred[n:,1:]@h.T-A[n:])),5)
 print(name,stats(D(x)))
 for nam2,w in [('B',B),('P',P),('R',R),('z',z)]:
  cross=(x[:,:,None]*w[:,None,:]).reshape(len(x),-1) if x.shape[1]<6 else np.sum(x[:,-4:]*w,axis=1)[:,None]
  base=D(x,w); dot=np.sum(x[:,-4:]*w,axis=1); print('  with',nam2,'dot',stats(D(base,dot)), 'cross',stats(D(base,cross)))
