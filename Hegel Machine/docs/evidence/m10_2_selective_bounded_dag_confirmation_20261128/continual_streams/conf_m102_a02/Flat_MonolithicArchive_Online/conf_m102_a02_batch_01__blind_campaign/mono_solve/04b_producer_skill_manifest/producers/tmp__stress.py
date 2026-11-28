import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
from batch_solver import solve_batch,_calibration,_time_basis
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; z=tr['traj']; t=s['eval_times']; b,d=_calibration(s['panel_a']); X=(x-b)/d
for j in range(12):
    q=np.arange(72) if j==0 else np.random.default_rng(j).permutation(72)
    a,c=q[:52],q[52:]; o=solve_batch(dict(s, train={k:v[a] for k,v in tr.items() if k!='summary'},test={'x':x[c]})); re=abs(o['y_hat']-y[c]); rt=abs(o['state_pred']-z[c]); extreme=np.max(abs(X[c]),axis=1)>=1.; print('fold',j,'y',round(re.mean(),5),'traj',round(rt.mean(),5),'worst timestamp',round(rt.mean(0).max(),5),'extreme',np.round([re[extreme].mean(),rt[extreme].mean()],5))
np.set_printoptions(precision=4,suppress=True,linewidth=160)
for name,J,n in [('main',X,52),('b',s['panel_b']['u'],19),('c',s['panel_c']['u'],19)]:
 K={'main':np.c_[y,z@np.linalg.pinv(_time_basis(t,z[:52]).T)],'b':s['panel_b']['v'],'c':s['panel_c']['v1']}[name]; i,j=np.triu_indices(4,1)
 for fn,f in [('cubic',lambda x:x**3),('tanh',lambda x:np.tanh(1.25*x)),('sine',lambda x:np.sin(2.9*x)),('rational',lambda x:x**3/(1+.22*x*x)),('tanhc',lambda x:np.tanh(1.25*x**3)),('mixed',lambda x:x*x*np.sin(1.25*x)),('signed_clip',lambda x:np.clip(x,-.65,.65))]:
    F=np.c_[np.ones(len(J)),J,J**2,f(J)]; C=np.linalg.lstsq(F[:n],K[:n],rcond=None)[0]; R=F@C-K; print(name,fn,np.round(np.mean(abs(R[:n]),0),4),np.round(np.mean(abs(R[n:]),0),4))
    if name=='main' and fn=='sine': print('coefficients',C, 'range',J.min(0),J.max(0)); print('resid correlation',np.corrcoef(np.c_[R,J[:,i]*J[:,j]].T)[:3,3:])
