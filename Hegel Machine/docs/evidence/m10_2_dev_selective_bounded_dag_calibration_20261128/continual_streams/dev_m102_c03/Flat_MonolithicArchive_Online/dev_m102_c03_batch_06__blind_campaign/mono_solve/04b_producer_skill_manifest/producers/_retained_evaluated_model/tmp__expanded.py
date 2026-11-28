import pickle,numpy as np,runpy
from itertools import combinations
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=52; t=s['eval_times']; fit=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-9)[0]; aff=lambda x:np.c_[np.ones(len(x)),x]; cyc=lambda x:x*np.roll(x,-1,axis=1)
z=runpy.run_path('/tmp/hegel_continual_archive/dev_m102_c03_batch_05_solver.py')['_calibrate'](s['panel_a'],tr['x']); pb=s['panel_b']; B=fit(aff(pb['u']),pb['v']); p=s['panel_c']; C=fit(aff(cyc(p['u'])),p['v1']); D=fit(aff(np.cos(2.4*p['u'])),p['v2']); print('B singulars',np.linalg.svd(B[1:],compute_uv=False),'summary',tr['summary'])
Q=-np.expm1(-t[:,None]*[.3954,1.8076,4.6966]); T=tr['traj']; A=fit(Q,T.T).T; Y=np.c_[tr['y'],A]; print('amplitude std',Y.std(0),'corr',np.corrcoef(Y.T)); print('first',np.c_[z,Y][:12].round(3))
base={'z':z,'raw':tr['x'],'B':aff(z)@B,'Bi':(z-B[0])@np.linalg.inv(B[1:]),'BT':z@B[1:].T+B[0]};
for k,q in list(base.items()):
 for f in [.8,1.25]:base[k+str(f)]=q*f
 for f in [1.,1.25]:base[k+'clip'+str(f)]=np.clip(q,-f,f);base[k+'tanh'+str(f)]=np.tanh(q*f)

def metrics(X):
 M=aff(X); P=M@fit(M[:n],Y[:n]); e=abs(Y-P); return np.r_[e[:n].mean(0),e[n:].mean(0),np.mean(abs(T[n:]-P[n:,1:]@Q.T))]
R=[]
for k,q in base.items():
 c=aff(cyc(q))@C; d=aff(np.cos(2.4*q))@D; cd=c*d; outer=(c[:,:,None]*d[:,None,:]).reshape(len(q),-1); diag=cd.sum(1); terms={'c,d':np.c_[c,d],'dot':np.c_[c,d,diag],'diag':np.c_[c,d,cd],'outer':np.c_[c,d,outer],'z+diag':np.c_[q,c,d,cd],'z+dot':np.c_[q,c,d,diag],'sq+diag':np.c_[q,q*q,c,d,cd],'cosprod':np.c_[c,d,cd,np.cos(2.4*cyc(q))],'sat':np.c_[c,d,cd,np.tanh(c),np.tanh(d)],'c/d':np.c_[c,d,cd,c/(1+.22*d*d),d/(1+.22*c*c)],'out/dot':np.c_[c,d,cd,c/(1+.22*diag[:,None]**2),d/(1+.22*diag[:,None]**2)]}
 for j,F in terms.items():R.append((k,j,metrics(F)))
for j in [0,1,2,3,4,5,6,7,8]: print('best',j,[(k,a,np.round(m,4)) for k,a,m in sorted(R,key=lambda x:x[2][j])[:4]])
np.savez('/tmp/expanded.npz',Y=Y,Q=Q,C=C,D=D,Bcoef=B,**base)
