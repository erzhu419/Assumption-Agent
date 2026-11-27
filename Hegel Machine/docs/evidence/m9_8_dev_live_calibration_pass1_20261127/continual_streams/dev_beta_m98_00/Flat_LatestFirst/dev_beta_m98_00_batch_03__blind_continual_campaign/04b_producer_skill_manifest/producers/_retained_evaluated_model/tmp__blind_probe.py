exec(open('/tmp/periodic_probe.py').read().split("for a in ('x','y'):")[0])
from scipy.optimize import lsq_linear
pulse_rhs=rhs.copy(); rhs=src.transpose(0,2,1).reshape(len(src),-1).T
obs=cal(s['y_obs_raw']); Z=s['Z_nuisance']; ridge=s['source_ridge']; box=s['u_box']; budget=s['total_budget']; print('QP',ridge,box,budget,s['nuisance_ridge'],'Z SVD',np.linalg.svd(Z,compute_uv=False),'response mean by shift',obs.reshape(3,-1).mean(1))
def fit(S,mask,yr,rz):
 z=Z[mask]; A=S[mask]; b=obs[mask]; p=np.linalg.solve(z.T@z+rz*np.eye(z.shape[1]),z.T); w=m._weights(A-z@(p@A),b-z@(p@b),yr,box,budget); g=p@(b-A@w); return obs-S@w-Z@g,w,g
wallsets={'open':x0,'periodic':np.r_[expected('x'),expected('y')],'reflect':np.tile([0.,1.,0.],4)}
for a in [0,0.25,0.5,0.75,1,1.25,1.5,2,3,4,6,10,30]:
 wallsets[f'robin{a}']=np.tile([a,1.,0.],4)
for j in [0,1]:
 for a in [0.,0.5,1.,2.,4.]:
  v=x0.copy(); v[6*j:6*(j+1)]=expected('xy'[j]); v[6*(1-j):6*(2-j):3]=a; wallsets[f'periodic{j}-robin{a}']=v
for key,x in wallsets.items():
 S=pred(x); sel=np.ones(len(obs),bool); e,w,g=fit(S,sel,1e-9,1e-9); er,wr,gr=fit(S,sel,ridge,s['nuisance_ridge']); cv=np.zeros(len(obs))
 for j in range(4):
  mask=(np.arange(len(obs))//4)%4==j; cv[mask]=fit(S,~mask,1e-9,1e-9)[0][mask]
 print(key, 'no ridge',np.sqrt(np.mean(e**2)), 'specified',np.sqrt(np.mean(er**2)), 'cv',np.sqrt(np.mean(cv.reshape(3,-1)**2,axis=1)), 'nuisance',g.round(3), 'w',w.round(3))
print('pulse budgets',[(p['w_pulse'].sum(),p['w_pulse'].min(),p['w_pulse'].max()) for p in pulses])
