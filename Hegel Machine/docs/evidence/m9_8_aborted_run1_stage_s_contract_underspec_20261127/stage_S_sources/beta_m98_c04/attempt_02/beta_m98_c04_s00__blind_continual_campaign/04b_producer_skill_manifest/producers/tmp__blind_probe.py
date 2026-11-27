from transport_probe import *
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
P=Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
R=np.eye(len(Z))-P
y=inv(s['y_obs_raw']); yp=R@y
rp=[]; boundary_h=[]; ad=[]
for ax in 'xy':
 h=s['L'+ax]/(s['n'+ax]-1); v=s['vel_'+ax][[0,-1]]
 rp+= [s['robin_'+ax]]*2; boundary_h += [h]*2; ad.extend(np.maximum(v*[1,-1],0)/h)
rp=np.array(rp); boundary_h=np.array(boundary_h); ad=np.array(ad)
from transport_probe import __dict__ as probe
probe['rhs']=np.column_stack([u.ravel(order='F') for u in s['source_modes']]); nom=rp/boundary_h-ad
pulsefit=least_squares(lambda b:(pred(1,'arith','local',b)@w.T-obs).ravel(),nom,gtol=1e-13).x
print('effective Robin',((pulsefit+ad)*boundary_h).tolist(), 'nominal',rp.tolist())
print('nuisance-free pulse residuals by pulse, mean/std',np.mean(pred(1,'arith','local',pulsefit)@w.T-obs,axis=0),np.std(pred(1,'arith','local',pulsefit)@w.T-obs,axis=0))
for b in [nom,pulsefit]:
 X=R@pred(1,'arith','local',b); print('S projected sv',np.linalg.svd(X,compute_uv=False));
 for ridge in [0,s['source_ridge']]:
  H=X.T@X+ridge*np.eye(len(w[0])); c=X.T@yp
  res=minimize(lambda q:(q@H@q/2-q@c,H@q-c),np.full(len(c),s['total_budget']/len(c)),jac=True,method='SLSQP',bounds=[(0,s['u_box'])]*len(c),constraints=[{'type':'eq','fun':lambda q:sum(q)-s['total_budget'],'jac':lambda q:np.ones(len(c))}],options={'ftol':1e-13}); print('nom/pulse',b,'ridge',ridge,'weights',res.x.round(6),'rmse',np.mean((X@res.x-yp)**2)**.5)
for reg in [0,0.01,0.05,0.2]:
 # Hold out complete sensor groups across shifts to diagnose blind correction.
 X0=pred(1,'arith','local',nom); out=[]
 for test in [np.zeros(len(Z),bool)]+[np.tile(np.arange(len(idx))%4==i,3) for i in range(4)]:
  tr=~test; zz=Z[tr]; Zinv=np.linalg.solve(zz.T@zz+0.000001*np.eye(zz.shape[1]),zz.T); test=test if np.any(test) else tr
  def fun(q):
   bb=nom+np.repeat(q[-2:],2)/boundary_h; residual=(pred(1,'arith','local',bb)@q[:-2]-y); g=Zinv@residual[tr]; a=residual[tr]-zz@g; return (a@a+reg*(q[-2:]@q[-2:]))/2
  b0=np.r_[np.full(w.shape[1],s['total_budget']/w.shape[1]),0,0]; eq={'type':'eq','fun':lambda q:sum(q[:-2])-s['total_budget']}; r=minimize(fun,b0,method='SLSQP',constraints=[eq],bounds=[(0,s['u_box'])]*w.shape[1]+[(-.45,.6)]*2,options={'ftol':1e-12,'maxiter':120}); residual=pred(1,'arith','local',nom+np.repeat(r.x[-2:],2)/boundary_h)@r.x[:-2]-y; out.append((r.x[-2:].round(5).tolist(),float(np.mean((residual[test]-Z[test]@(Zinv@residual[tr]))**2)**.5)))
 print('blind correction regularizer',reg,'full and held-out:',out)
print('calibration row sparsity',np.unique(np.count_nonzero(x,axis=1),return_counts=True)); print('max affine residual',np.max(abs(np.c_[x,np.ones(len(x))]@cf-y),axis=0)); print('paired sweep span',np.ptp(x,axis=0)); print('bank sweep residual versus affine',np.std(cal['single_bank_sweeps']['raw_readout']-cal['single_bank_sweeps']['ref_signal']))
